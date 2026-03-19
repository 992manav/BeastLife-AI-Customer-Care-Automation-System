import pandas as pd

files = {
    'MASTER_DATASET_UNIFIED.csv': 'dataset/MASTER_DATASET_UNIFIED.csv',
    'BEASTLIFE_QUERIES_CATEGORIZED.csv': 'dataset/BEASTLIFE_QUERIES_CATEGORIZED.csv',
    'BEASTLIFE_ALL_QUERIES_COMBINED.csv': 'dataset/BEASTLIFE_ALL_QUERIES_COMBINED.csv',
    'ALL_CSV_COMBINED_MASTER.csv': 'dataset/ALL_CSV_COMBINED_MASTER.csv',
}

print("=" * 80)
print("DATASET VERIFICATION")
print("=" * 80)

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"\n{name}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    if 'Category' in df.columns:
        print(f"  Categories: {df['Category'].nunique()} unique")

print("\n" + "=" * 80)
