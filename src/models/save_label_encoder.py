# src/models/save_label_encoder.py
# Loads train.csv, builds LabelEncoder, saves classes_ to models/label_encoder.npy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np, os
df = pd.read_csv("data/train.csv")
le = LabelEncoder().fit(df["family_code"])
os.makedirs("models", exist_ok=True)
np.save("models/label_encoder.npy", le.classes_)
print("Saved label encoder classes to models/label_encoder.npy")
