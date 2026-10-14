import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, time, math, json, glob
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import shapely
from shapely.ops import unary_union
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from common import *
from model import HydroNet, UncertaintyWeighting


def infinite(loader):
    while True:
        for b in loader:
            yield b

def gradnorm(module, loss):
    ps = [p for p in module.parameters() if p.requires_grad]
    g = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))

@torch.no_grad()
def eval_wells(model, wdf, store, norm, seqtab, dev, workers=0):
    dl = DataLoader(WellDataset(wdf, store, norm, seqtab), batch_size=64, num_workers=workers)
    model.eval(); out = []
    for x, s, _ in dl:
        h = model.embed(x.to(dev), s.to(dev))
        z, sl = model.well_forward(h)
        out.append(torch.stack([z, sl, model.mil_head(h).squeeze(-1)], 1).cpu())
    o = torch.cat(out).numpy()
    r = wdf.copy().reset_index(drop=True)
    r["z_pred"], r["stress_logit"], r["mil_inst"] = o[:, 0], o[:, 1], o[:, 2]
    return r

def well_metrics(r):
    out = {}
    m = r.regmask > 0
    if m.sum() > 5:
        out["rmse_z"] = float(np.sqrt(((r.z_pred[m] - r.z[m]) ** 2).mean()))
        out["spearman_z"] = float(spearmanr(r.z_pred[m], r.z[m])[0])
    c = r[r.clsmask > 0]
    if c.stressed.nunique() == 2:
        out["auc_stress"] = float(roc_auc_score(c.stressed, c.stress_logit))
        out["ap_stress"] = float(average_precision_score(c.stressed, c.stress_logit))
        out["bal_acc_stress"] = float(balanced_accuracy_score(c.stressed, c.stress_logit > 0))
    return out

@torch.no_grad()
def infer_cells(model, keys, years, cells, store, norm, seqtab, dev, workers, min_valid=0.3):
    rows = pd.concat([pd.DataFrame({"taluk": t, "year": y, "row": cells[t][:, 0], "col": cells[t][:, 1]})
                      for t in keys for y in years], ignore_index=True)
    dl = DataLoader(GridDataset(rows, store, norm, seqtab), batch_size=256, num_workers=workers)
    model.eval(); H, VF = [], []
    for x, s in dl:
        VF.append(x[:, -1].mean((1, 2)).numpy())
        H.append(model.embed(x.to(dev), s.to(dev)))
    H = torch.cat(H)
    rows["valid_frac"] = np.concatenate(VF)
    cell_parts, tal_parts = [], []
    for (t, y), g in rows[rows.valid_frac >= min_valid].groupby(["taluk", "year"]):
        Hg = H[torch.from_numpy(g.index.values).to(dev)]
        bag, att, inst = model.mil_forward(Hg)
        z, sl = model.well_forward(Hg)
        ent = float(-(att * att.clamp_min(1e-12).log()).sum() / math.log(max(len(att), 2)))
        c = g.copy()
        c["attn"], c["inst_logit"] = att.cpu().numpy(), inst.cpu().numpy()
        c["z_pred"], c["stress_logit"] = z.cpu().numpy(), sl.cpu().numpy()
        cell_parts.append(c)
        tal_parts.append(dict(taluk=t, year=int(y), bag_logit=float(bag), attn_entropy=ent,
                              mean_inst=float(inst.mean()), n_cells=len(g)))
    return pd.concat(cell_parts), pd.DataFrame(tal_parts)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="all")
    ap.add_argument("--tag", default="joint_uw")
    ap.add_argument("--mode", default="joint", choices=["joint", "well", "mil"])
    ap.add_argument("--weighting", default="uncertainty", choices=["uncertainty", "fixed"])
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--no_sar", action="store_true")
    ap.add_argument("--no_temporal", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--wbs", type=int, default=32)
    ap.add_argument("--bags", type=int, default=2)
    ap.add_argument("--bag_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--enc_mult", type=float, default=0.3)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--freeze", type=int, default=3)
    ap.add_argument("--mil_years", default="2023,2024,2025")
    ap.add_argument("--infer_years", default="2025")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_semi", action="store_true")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--buffer_km", type=float, default=3.0)
    ap.add_argument("--perm", action="store_true")
    return ap.parse_args()

def run_fold(a, k, tal, wells, cells, store, norm, seqtab, dev, out):
    torch.manual_seed(a.seed + k); np.random.seed(a.seed + k)
    train_t = set(tal[(tal.fold != k) & (tal.fold >= 0)].taluk)
    val_t = set(tal[tal.fold == k].taluk)
    semi_t = set(tal[tal.fold == -1].taluk)
    lab = {t: int(l) for t, l in zip(tal.taluk, tal.label) if t in train_t and l in (0, 1)}
    wtr, wva = wells[wells.taluk.isin(train_t)], wells[wells.taluk.isin(val_t)]

    # ---- leakage guard: drop training cells/wells within buffer_km of any validation taluk ----
    cells_tr = {t: cells[t] for t in lab}
    if a.buffer_km > 0:
        root = store._root()
        zone = unary_union([taluk_geoms()[t] for t in val_t]).buffer(a.buffer_km * 1000)
        n_cells0 = sum(len(c) for c in cells_tr.values())
        for t in list(cells_tr):
            c = cells_tr[t]
            x, y = rc_to_xy(root, [t] * len(c), c[:, 0], c[:, 1])
            cells_tr[t] = c[~shapely.contains_xy(zone, x, y)]
        x, y = rc_to_xy(root, wtr.taluk.values, wtr.row.values, wtr.col.values)
        n_w0 = len(wtr); wtr = wtr[~shapely.contains_xy(zone, x, y)]
        lab = {t: l for t, l in lab.items() if len(cells_tr[t]) >= a.bag_size}
        print(f"[fold {k}] buffer {a.buffer_km} km removed {n_cells0 - sum(len(c) for c in cells_tr.values())} "
              f"cells and {n_w0 - len(wtr)} train wells")
    if a.mode != "well":
        assert len(set(lab.values())) == 2, "need both OE and safe taluks in the training set"
    if a.perm:
        rng_p = np.random.default_rng(1000 + a.seed + k)
        ks = list(lab)
        lab = dict(zip(ks, rng_p.permutation([lab[t] for t in ks]).tolist()))
    print(f"[fold {k}] train taluks {len(lab)} (OE {sum(lab.values())}, safe {len(lab)-sum(lab.values())}), "
          f"val taluks {len(val_t)}, train wells {len(wtr)}, val wells {len(wva)}")

    model = HydroNet(len(store.out_names) + 1, len(store.seq_names), not a.no_temporal, a.width).to(dev)
    pre = bool(a.pretrained)
    if pre:
        model.enc.load_state_dict(torch.load(a.pretrained, map_location=dev))
    enc_ids = {id(p) for p in model.enc.parameters()}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    uw = UncertaintyWeighting(3).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": a.lr * (a.enc_mult if pre else 1.0)},
        {"params": other, "lr": a.lr},
        {"params": uw.parameters(), "lr": 1e-3, "weight_decay": 0.0}], weight_decay=1e-2)
    base_lrs = [g["lr"] for g in opt.param_groups]

    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0)
    well_it = infinite(DataLoader(WellDataset(wtr, store, norm, seqtab, True), batch_size=a.wbs,
                                  shuffle=True, drop_last=True, **kw))
    bag_it = None
    if a.mode != "well":
        bag_ds = BagDataset({t: cells_tr[t] for t in lab}, lab, [int(y) for y in a.mil_years.split(",")],
                            store, norm, seqtab, a.bag_size)
        bag_it = infinite(DataLoader(bag_ds, batch_size=a.bags, **kw))
    npos, nneg = wtr.stressed[wtr.clsmask > 0].sum(), (1 - wtr.stressed[wtr.clsmask > 0]).sum()
    pos_w = torch.tensor(float(nneg / max(npos, 1)), device=dev)
    wprobe = wtr.sample(min(200, len(wtr)), random_state=0)

    log = []
    for ep in range(a.epochs):
        lr_f = min(1.0, (ep + 1) / 2) * 0.5 * (1 + math.cos(math.pi * ep / a.epochs))
        for g, b in zip(opt.param_groups[:2], base_lrs[:2]):
            g["lr"] = b * lr_f
        frozen = pre and ep < a.freeze
        for p in model.enc.parameters():
            p.requires_grad = not frozen
        mil_on = a.mode != "well" and (a.mode == "mil" or ep >= a.warm)
        lam = 1.0 if a.mode == "mil" else min(1.0, max(0.0, (ep - a.warm + 1) / max(a.ramp, 1)))
        well_on = a.mode != "mil"
        model.train(); agg = {"reg": 0, "cls": 0, "mil": 0}; ents = []; gn = (np.nan, np.nan); t0 = time.time()
        for step in range(a.steps):
            losses, ids = [], []
            if well_on:
                x, s, y = [t.to(dev) for t in next(well_it)]
                z, sl = model.well_forward(model.embed(x, s))
                L_reg = (F.huber_loss(z, y[:, 0], reduction="none") * y[:, 2]).sum() / y[:, 2].sum().clamp(min=1)
                L_cls = (F.binary_cross_entropy_with_logits(sl, y[:, 1], pos_weight=pos_w, reduction="none")
                         * y[:, 3]).sum() / y[:, 3].sum().clamp(min=1)
                losses += [L_reg, L_cls]; ids += [0, 1]
                agg["reg"] += L_reg.item(); agg["cls"] += L_cls.item()
            if mil_on:
                xb, sb, lb = [t.to(dev) for t in next(bag_it)]
                B, n = xb.shape[:2]
                hb = model.embed(xb.flatten(0, 1), sb.flatten(0, 1)).view(B, n, -1)
                outs = [model.mil_forward(hb[i]) for i in range(B)]
                L_mil = F.binary_cross_entropy_with_logits(torch.stack([o[0] for o in outs]), lb)
                ents += [float(-(o[1].detach() * o[1].detach().clamp_min(1e-12).log()).sum() / math.log(n)) for o in outs]
                losses.append(lam * L_mil); ids.append(2); agg["mil"] += L_mil.item()
                if a.mode == "joint" and step % 50 == 0 and not frozen:
                    gn = (gradnorm(model.enc, L_reg + L_cls), gradnorm(model.enc, L_mil))
            total = uw(losses, ids) if a.weighting == "uncertainty" else sum(losses)
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        rec = {"fold": k, "epoch": ep + 1, "reg": agg["reg"] / a.steps, "cls": agg["cls"] / a.steps,
               "mil": agg["mil"] / a.steps, "lambda_mil": lam,
               "attn_entropy": float(np.mean(ents)) if ents else np.nan,
               "gn_well": gn[0], "gn_mil": gn[1], "s": uw.s.detach().cpu().numpy().round(3).tolist(),
               "sec": round(time.time() - t0)}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            rec.update({"val_" + kk: v for kk, v in well_metrics(eval_wells(model, wva, store, norm, seqtab, dev)).items()})
            rec.update({"train_" + kk: v for kk, v in well_metrics(eval_wells(model, wprobe, store, norm, seqtab, dev)).items()})
        log.append(rec); print(rec)
    pd.DataFrame(log).to_csv(f"{out}/log_fold{k}.csv", index=False)

    # ---- out-of-fold outputs ----
    wo = eval_wells(model, wva, store, norm, seqtab, dev); wo["fold"] = k
    wo.to_csv(f"{out}/oof_wells_fold{k}.csv", index=False)
    role = {t: "val" for t in val_t}
    if not a.skip_semi:
        role.update({t: "semi" for t in semi_t})
    years = [int(v) for v in a.infer_years.split(",")]
    cdf, tdf = infer_cells(model, sorted(role), years, cells, store, norm, seqtab, dev, a.workers)
    cdf["role"], cdf["model_fold"] = cdf.taluk.map(role), k
    tdf["role"], tdf["model_fold"] = tdf.taluk.map(role), k
    tdf["label"] = tdf.taluk.map(tal.set_index("taluk").label)
    tdf.to_csv(f"{out}/oof_taluks_fold{k}.csv", index=False)
    cdf.to_parquet(f"{out}/oof_cells_fold{k}.parquet")
    torch.save(model.state_dict(), f"{out}/model_fold{k}.pt")
    return well_metrics(wo)


def bootstrap_auc(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(n):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])

def summarize(out):
    T = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out}/oof_taluks_fold*.csv")])
    W = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out}/oof_wells_fold*.csv")])
    res = {}
    v = T[(T.role == "val") & (T.label.isin([0, 1]))].groupby(["taluk", "label"], as_index=False).bag_logit.mean()
    if v.label.nunique() == 2:
        res["taluk_auc_pooled_oof"] = float(roc_auc_score(v.label, v.bag_logit))
        res["taluk_auc_95ci"] = bootstrap_auc(v.label, v.bag_logit).round(3).tolist()
        res["n_oe_safe"] = [int((v.label == 1).sum()), int((v.label == 0).sum())]
        res["mean_logit_safe_vs_oe"] = [float(v[v.label == 0].bag_logit.mean()), float(v[v.label == 1].bag_logit.mean())]
        pred = (v.bag_logit > 0).astype(int)
        res["taluk_bal_acc_at_0"] = float(balanced_accuracy_score(v.label, pred))
        res["taluk_acc_at_0"] = float((pred == v.label).mean())
        res["taluk_majority_baseline_acc"] = float(max(v.label.mean(), 1 - v.label.mean()))
    s = T[T.role == "semi"].groupby("taluk").bag_logit.mean()
    if len(s):
        res["semi_critical_mean_logit"] = float(s.mean())
    res.update({"wells_" + k: v for k, v in well_metrics(W).items()})
    c = W[W.clsmask > 0]
    res["wells_stress_prevalence_baseline_ap"] = float(c.stressed.mean())
    W["mil_c"] = W.mil_inst - W.groupby("taluk").mil_inst.transform("mean")
    W["z_c"] = W.z - W.groupby("taluk").z.transform("mean")
    m = W.regmask > 0
    if m.sum() > 10:
        res["subtaluk_spearman_mil_vs_wellz"] = float(spearmanr(W.mil_c[m], W.z_c[m])[0])
        res["subtaluk_spearman_wellhead_vs_wellz"] = float(spearmanr(W.z_pred[m], W.z[m])[0])
    json.dump(res, open(f"{out}/summary.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED for training: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: No GPU detected, using CPU. This will be very slow!")
        
    a = parse()
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    out = f"runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f"{out}/args.json", "w"), indent=2)
    store = PatchStore(not a.no_sar); norm = Normalizer(store.out_names); seqtab = SeqTable(not a.no_sar)
    df = load_index()
    tal = df.groupby("taluk").agg(fold=("fold", "first"), label=("label", "first")).reset_index()
    wells = df[df.is_well].copy()
    g = wells.groupby("taluk")["gwl"]
    sd, n = g.transform("std"), g.transform("count")
    wells["z"] = (wells.gwl - g.transform("mean")) / sd
    wells["regmask"] = ((n >= 5) & (sd > 1e-6) & wells.z.notna()).astype(float)
    wells["z"] = wells.z.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-3, 3)
    wells["clsmask"] = wells.stressed.notna().astype(float)
    wells["stressed"] = wells.stressed.fillna(0).astype(float)
    grid = df[~df.is_well]
    cells = {t: gg[["row", "col"]].drop_duplicates().values for t, gg in grid.groupby("taluk")}
    folds = sorted(f for f in tal.fold.unique() if f >= 0) if a.fold == "all" else [int(a.fold)]
    for k in folds:
        print("fold", k, run_fold(a, k, tal, wells, cells, store, norm, seqtab, dev, out))
    if a.fold == "all":
        summarize(out)

if __name__ == "__main__":
    main()
