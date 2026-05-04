"""
US_Accidents Dataset Inspector
Efficiently inspects a 3GB CSV without loading it fully into memory.
Outputs: headers, dtypes, sample data, null counts, value distributions, and ML suitability assessment.
"""
import pandas as pd
import numpy as np
import sys
import os
import json

CSV_PATH = "/Users/prags/Desktop/MLPROJECT/data/US_Accidents.csv"
OUTPUT_PATH = "/Users/prags/Desktop/MLPROJECT/outputs/dataset_inspection.json"

print("=" * 80)
print("US ACCIDENTS DATASET INSPECTION")
print("=" * 80)

# ── 1. Read just the header row ──
print("\n[1/7] Reading headers...")
header_df = pd.read_csv(CSV_PATH, nrows=0)
columns = list(header_df.columns)
print(f"Total columns: {len(columns)}")
print(f"Columns:\n  {columns}")

# ── 2. Count total rows efficiently (without loading data) ──
print("\n[2/7] Counting total rows (streaming)...")
chunk_size = 500_000
total_rows = 0
for chunk in pd.read_csv(CSV_PATH, chunksize=chunk_size, usecols=[0]):
    total_rows += len(chunk)
print(f"Total rows: {total_rows:,}")

# ── 3. Read a sample (first 50,000 rows) for detailed inspection ──
print("\n[3/7] Loading first 50,000 rows for detailed inspection...")
sample_df = pd.read_csv(CSV_PATH, nrows=50_000, low_memory=False)

print("\n--- Data Types ---")
for col in sample_df.columns:
    print(f"  {col:30s} → {str(sample_df[col].dtype):15s}")

# ── 4. Null analysis (full dataset via streaming) ──
print("\n[4/7] Analyzing nulls across FULL dataset (streaming)...")
null_counts = pd.Series(0, index=columns, dtype='int64')
for chunk in pd.read_csv(CSV_PATH, chunksize=chunk_size, low_memory=False):
    null_counts += chunk.isnull().sum()

print("\n--- Null Counts (Full Dataset) ---")
null_pct = (null_counts / total_rows * 100).round(2)
for col in columns:
    status = "✅" if null_pct[col] < 1 else ("⚠️" if null_pct[col] < 10 else "🔴")
    print(f"  {status} {col:30s} → {null_counts[col]:>10,} nulls ({null_pct[col]:>6.2f}%)")

# ── 5. Sample data from each column ──
print("\n[5/7] Sample values (first 5 non-null) per column...")
for col in sample_df.columns:
    unique_sample = sample_df[col].dropna().unique()[:5]
    print(f"  {col:30s} → {list(unique_sample)}")

# ── 6. Value distributions for key columns ──
print("\n[6/7] Value distributions for key columns...")

# Target variable: Severity
print("\n--- Severity Distribution (full dataset, streaming) ---")
severity_counts = pd.Series(dtype='int64')
for chunk in pd.read_csv(CSV_PATH, chunksize=chunk_size, usecols=['Severity'], low_memory=False):
    severity_counts = severity_counts.add(chunk['Severity'].value_counts(), fill_value=0)
severity_counts = severity_counts.astype(int).sort_index()
for sev, count in severity_counts.items():
    pct = count / total_rows * 100
    print(f"  Severity {sev}: {count:>10,} ({pct:.2f}%)")

# Categorical columns distributions (from sample)
cat_cols = ['Weather_Condition', 'Sunrise_Sunset', 'State', 'Side', 'Source', 'Country']
existing_cat_cols = [c for c in cat_cols if c in sample_df.columns]

for col in existing_cat_cols:
    print(f"\n--- {col} (top 10 from sample) ---")
    vc = sample_df[col].value_counts().head(10)
    for val, cnt in vc.items():
        print(f"  {str(val):35s} → {cnt:>6,}")

# Boolean columns
bool_cols = [c for c in sample_df.columns if sample_df[c].dtype == 'bool' or 
             set(sample_df[c].dropna().unique()).issubset({True, False, 'True', 'False'})]
if bool_cols:
    print(f"\n--- Boolean Feature Columns ---")
    for col in bool_cols:
        vc = sample_df[col].value_counts()
        true_pct = (vc.get(True, 0) + vc.get('True', 0)) / len(sample_df) * 100
        print(f"  {col:30s} → True: {true_pct:.1f}%")

# Numeric distributions (from sample)
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n--- Numeric Feature Statistics (from 50k sample) ---")
stats = sample_df[numeric_cols].describe().T
stats['null_pct'] = sample_df[numeric_cols].isnull().sum() / len(sample_df) * 100
print(stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'null_pct']].to_string())

# ── 7. ML Suitability Assessment ──
print("\n" + "=" * 80)
print("[7/7] ML SUITABILITY ASSESSMENT")
print("=" * 80)

# Class imbalance
print("\n🎯 TARGET VARIABLE (Severity):")
for sev, count in severity_counts.items():
    pct = count / total_rows * 100
    bar = "█" * int(pct / 2)
    print(f"  Class {sev}: {count:>10,} ({pct:>5.2f}%) {bar}")

# Identify high-null columns
high_null = null_pct[null_pct > 30].sort_values(ascending=False)
if len(high_null) > 0:
    print(f"\n⚠️  COLUMNS WITH >30% NULLS (candidates for dropping):")
    for col, pct_val in high_null.items():
        print(f"  - {col}: {pct_val:.1f}% missing")

moderate_null = null_pct[(null_pct > 5) & (null_pct <= 30)].sort_values(ascending=False)
if len(moderate_null) > 0:
    print(f"\n⚠️  COLUMNS WITH 5-30% NULLS (need imputation):")
    for col, pct_val in moderate_null.items():
        print(f"  - {col}: {pct_val:.1f}% missing")

# Feature type summary
print(f"\n📊 FEATURE TYPE SUMMARY:")
print(f"  Numeric columns:     {len(numeric_cols)}")
cat_count = len(sample_df.select_dtypes(include=['object']).columns)
bool_count = len(bool_cols)
print(f"  Categorical columns: {cat_count}")
print(f"  Boolean columns:     {bool_count}")
print(f"  Total:               {len(columns)}")

# Unique value counts for categoricals (cardinality check)
print(f"\n📊 CATEGORICAL CARDINALITY (from sample):")
for col in sample_df.select_dtypes(include=['object']).columns:
    nunique = sample_df[col].nunique()
    status = "✅" if nunique < 50 else ("⚠️ HIGH" if nunique < 500 else "🔴 VERY HIGH")
    print(f"  {col:30s} → {nunique:>6,} unique values  {status}")

print("\n✅ Inspection complete!")
print(f"Total rows: {total_rows:,} | Total columns: {len(columns)}")
