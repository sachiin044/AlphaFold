# File: src/data/dataset_builder.py

from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import re

fasta_file = r"src\data\astral-scopedom-seqres-gd-all-2.07-stable.fa"
output_dir = "data"
TOP_N_FAMILIES = 50
MIN_SAMPLES = 5

os.makedirs(output_dir, exist_ok=True)

records = []
for record in SeqIO.parse(fasta_file, "fasta"):
    header = record.description
    domain_id = record.id.split()[0]
    match = re.search(r"\s([a-z]\.\d+\.\d+\.\d+)\s", header)
    if match:
        family_code = match.group(1)
        records.append((domain_id, str(record.seq), family_code))

df = pd.DataFrame(records, columns=["domain_id", "sequence", "family_code"])
print(f"Parsed {len(df)} sequences with family codes from FASTA.")

# Filter families with enough samples
fam_counts = df["family_code"].value_counts()
valid_fams = fam_counts[fam_counts >= MIN_SAMPLES].index
TOP_N_FAMILIES = min(TOP_N_FAMILIES, len(valid_fams))
top_fams = fam_counts.loc[valid_fams].nlargest(TOP_N_FAMILIES).index

df = df[df["family_code"].isin(top_fams)]
print(f"Filtered to top {TOP_N_FAMILIES} families with ≥{MIN_SAMPLES} samples each: {len(df)} sequences remain.")

# Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["family_code"], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["family_code"], random_state=42)

train_df.to_csv(f"{output_dir}/train.csv", index=False)
val_df.to_csv(f"{output_dir}/val.csv", index=False)
test_df.to_csv(f"{output_dir}/test.csv", index=False)

# Label encoder
label_encoder = LabelEncoder()
label_encoder.fit(train_df["family_code"])
joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")

print("\n✅ Dataset building complete.")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
