import pandas as pd

df = pd.read_csv('data3.csv')

print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns.\n")
print("Available columns:", list(df.columns), "\n")

columns_to_show = ['sakid', 'tittel', 'url', 'emneord']

for col in columns_to_show:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: '{col}'")

print("Sample rows:\n")
print(df[columns_to_show].head(1).to_string(index=False))
