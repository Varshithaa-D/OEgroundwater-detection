import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, time, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from common import *
from model import Encoder, Decoder

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED for pretraining: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: No GPU detected, using CPU. This will be very slow!")
        
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_sar", action="store_true")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store = PatchStore(not a.no_sar); norm = Normalizer(store.out_names)
    df = load_index()
    g = df[(~df.is_well) & (df.year >= 2016)]
    g = g.sample(min(a.n, len(g)), random_state=0)
    dl = DataLoader(GridDataset(g, store, norm, None, augment=True), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, persistent_workers=a.workers > 0, drop_last=True)
    C = len(store.out_names)
    enc, dec = Encoder(C + 1, a.width).to(dev), Decoder(4 * a.width, C).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    
    print(f"Starting pretraining on {len(g)} samples for {a.epochs} epochs...")
    for ep in range(a.epochs):
        t0, tot, nb = time.time(), 0.0, 0
        for x, _ in dl:
            x = x.to(dev)
            m = F.interpolate((torch.rand(x.size(0), 1, 4, 4, device=dev) < 0.5).float(), scale_factor=8, mode="nearest")
            xin = x.clone(); xin[:, :C] = xin[:, :C] * (1 - m)
            f, _ = enc(xin); rec = dec(f)
            w = m * x[:, C:C + 1]
            loss = (((rec - x[:, :C]) ** 2) * w).sum() / (w.sum() * C + 1e-6)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"epoch {ep+1}/{a.epochs} loss {tot/nb:.4f} ({time.time()-t0:.0f}s)")
        torch.save(enc.state_dict(), f"pretrained_{'nosar' if a.no_sar else 'sar'}.pt")
