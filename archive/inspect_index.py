import pandas as pd
df = pd.read_parquet("index.parquet")
print(df.dtypes, "\n")
print(df.head(3).T, "\n")
print("rows:", len(df))
for c in df.columns:
    if df[c].nunique() < 15:
        print(c, df[c].value_counts(dropna=False).to_dict())
