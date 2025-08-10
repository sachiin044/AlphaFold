# src/models/infer_wrapper.py
"""
Inference wrapper for Protein Family Classification
Auto-detects model type (MLP or Conv) from checkpoint and runs prediction.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/protein_classifier.pt"

# --------------------------
# Model Definitions
# --------------------------
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# --------------------------
# ESM Embedding Function
# --------------------------
def compute_esm_embedding(seq, model_name="esm2_t12_35M_UR50D"):
    import esm
    model_func = getattr(esm.pretrained, model_name)
    model, alphabet = model_func()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    batch = [("seq1", seq.upper())]  # ESM expects uppercase
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    token_reps = results["representations"][12]
    return token_reps[0, 1:len(seq)+1].mean(0).cpu().numpy()

# --------------------------
# Load Checkpoint
# --------------------------
print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# Restore label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = checkpoint["label_encoder"]

# Detect model type
model_type = checkpoint.get("model_type", "mlp")
if model_type == "mlp":
    input_dim = checkpoint["input_dim"]
    model = MLPClassifier(input_dim=input_dim, hidden_dims=[1024, 512], num_classes=len(label_encoder.classes_))
elif model_type == "conv":
    vocab_size = 22  # 20 amino acids + pad + unknown
    model = ConvClassifier(vocab_size=vocab_size, embed_dim=128, num_classes=len(label_encoder.classes_))
else:
    raise ValueError(f"Unknown model type: {model_type}")

model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE).eval()

# --------------------------
# Prediction Functions
# --------------------------
def predict_sequence(seq):
    """Predict family code from raw protein sequence."""
    if model_type == "mlp":
        emb = compute_esm_embedding(seq)
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        outputs = model(emb_tensor)
    else:
        aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i+1 for i, aa in enumerate(aa_vocab)}
        encoded = [aa_to_idx.get(a.upper(), 0) for a in seq]
        MAX_SEQ_LEN = 512
        if len(encoded) > MAX_SEQ_LEN:
            encoded = encoded[:MAX_SEQ_LEN]
        else:
            encoded += [0] * (MAX_SEQ_LEN - len(encoded))
        seq_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(DEVICE)
        outputs = model(seq_tensor)

    pred_idx = torch.argmax(outputs, dim=1).item()
    return label_encoder.classes_[pred_idx]

# --------------------------
# CLI Entry Point
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/models/infer_wrapper.py <sequence>")
        sys.exit(1)

    seq = sys.argv[1]
    pred = predict_sequence(seq)
    print(f"Predicted Family: {pred}")
