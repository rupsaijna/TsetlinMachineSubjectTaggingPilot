import pandas as pd

# Load your CSV
df = pd.read_csv('data3.csv')

# Print basic info
print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns.\n")
print("📋 Available columns:", list(df.columns), "\n")

# Display selected columns
columns_to_show = ['sakid', 'tittel', 'url', 'emneord']

# Check if those columns exist
for col in columns_to_show:
    if col not in df.columns:
        raise ValueError(f"❌ Missing expected column: '{col}'")

# Print first 10 rows of selected columns
print("🔍 Sample rows:\n")
print(df[columns_to_show].head(1).to_string(index=False))
