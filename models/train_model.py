# src/models/train_model.py
"""
Improved Model Training Script for Protein Family Classification
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ===============================
# Step 1: Configuration
# ===============================
TRAIN_CSV = r"data\test.csv"
VAL_CSV = r"data\val.csv"
MODEL_SAVE_PATH = "models/protein_classifier.pt"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4  # lowered learning rate
MAX_SEQ_LEN = 512  # truncate/pad length

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ===============================
# Step 2: Dataset Class
# ===============================
class ProteinDataset(Dataset):
    def __init__(self, csv_file, label_encoder):
        self.df = pd.read_csv(csv_file)
        self.label_encoder = label_encoder
        self.sequences = self.df["sequence"].apply(self.encode_sequence).values
        self.labels = label_encoder.transform(self.df["family_code"])

    def encode_sequence(self, seq):
        aa_vocab = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard AAs
        aa_to_idx = {aa: i+2 for i, aa in enumerate(aa_vocab)}  # 0: padding, 1: unknown
        encoded = [aa_to_idx.get(a, 1) for a in seq]  # unknown -> 1
        if len(encoded) > MAX_SEQ_LEN:
            encoded = encoded[:MAX_SEQ_LEN]
        else:
            encoded += [0] * (MAX_SEQ_LEN - len(encoded))
        return np.array(encoded, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

# ===============================
# Step 3: Load Data & Labels
# ===============================
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# Print class distribution for imbalance check
print("Train class distribution:\n", train_df["family_code"].value_counts())
print("Val class distribution:\n", val_df["family_code"].value_counts())

label_encoder = LabelEncoder()
label_encoder.fit(train_df["family_code"])

train_dataset = ProteinDataset(TRAIN_CSV, label_encoder)
val_dataset = ProteinDataset(VAL_CSV, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# Step 4: Model Architecture
# ===============================
class ProteinClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(ProteinClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 256, batch_first=True, bidirectional=True)  # increased hidden size
        self.dropout = nn.Dropout(0.5)  # increased dropout
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[0], h[1]), dim=1)  # concat bidirectional hidden states
        x = self.dropout(h)
        return self.fc(x)

# ===============================
# Step 5: Prepare Loss & Optimizer
# ===============================
VOCAB_SIZE = 22  # 20 + padding(0) + unknown(1)
NUM_CLASSES = len(label_encoder.classes_)

# Calculate class weights for imbalance
class_counts = train_df["family_code"].value_counts().sort_index()
class_weights = 1.0 / class_counts.values
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

model = ProteinClassifier(vocab_size=VOCAB_SIZE, embed_dim=64, num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ===============================
# Step 6: Training Loop
# ===============================
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for batch_idx, (seqs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(seqs)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    scheduler.step()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            outputs = model(seqs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Val Acc: {val_acc:.4f}")

# ===============================
# Step 7: Save Model & Label Encoder
# ===============================
torch.save({
    "model_state": model.state_dict(),
    "label_encoder": label_encoder.classes_
}, MODEL_SAVE_PATH)

print(f"✅ Model saved to {MODEL_SAVE_PATH}")
