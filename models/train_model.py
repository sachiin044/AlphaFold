# src/models/train_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

TRAIN_CSV = r"data\train.csv"
VAL_CSV = r"data\val.csv"
MODEL_SAVE_PATH = r"models\protein_classifier.pt"
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class ProteinDataset(Dataset):
    def __init__(self, csv_file, label_encoder):
        self.df = pd.read_csv(csv_file)
        self.label_encoder = label_encoder
        self.sequences = []
        self.lengths = []
        for seq in self.df["sequence"]:
            enc = self.encode_sequence(seq)
            self.sequences.append(enc)
            self.lengths.append(min(len(seq), MAX_SEQ_LEN))  # <-- capped length
        self.sequences = np.array(self.sequences)
        self.lengths = np.array(self.lengths)
        self.labels = label_encoder.transform(self.df["family_code"])

    def encode_sequence(self, seq):
        aa_vocab = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i+1 for i, aa in enumerate(aa_vocab)}
        encoded = [aa_to_idx.get(a, 0) for a in seq]
        if len(encoded) > MAX_SEQ_LEN:
            encoded = encoded[:MAX_SEQ_LEN]
        else:
            encoded += [0] * (MAX_SEQ_LEN - len(encoded))
        return np.array(encoded, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.labels[idx]),
            torch.tensor(self.lengths[idx])
        )


class ProteinClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat((h[0], h[1]), dim=1)
        out = self.dropout(h)
        return self.fc(out)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

label_encoder = LabelEncoder()
label_encoder.fit(train_df["family_code"])

train_dataset = ProteinDataset(TRAIN_CSV, label_encoder)
val_dataset = ProteinDataset(VAL_CSV, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

VOCAB_SIZE = 21
NUM_CLASSES = len(label_encoder.classes_)

# Class weights
class_counts = train_df["family_code"].value_counts().sort_index()
class_weights = 1.0 / class_counts
weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(DEVICE)

model = ProteinClassifier(VOCAB_SIZE, 128, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for seqs, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        seqs, labels, lengths = seqs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels, lengths in val_loader:
            seqs, labels, lengths = seqs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            outputs = model(seqs, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Val Acc: {val_acc:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "label_encoder": label_encoder.classes_
}, MODEL_SAVE_PATH)

print(f"✅ Model saved to {MODEL_SAVE_PATH}")
