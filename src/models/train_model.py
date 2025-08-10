# File: src/models/train_model.py
"""
High-accuracy training pipeline for Protein Family Classification.
- Uses ESM embeddings (recommended) OR trains from augmented sequences.
- Mixed precision, AdamW, label smoothing, class weights, mixup, augmentation.
Do NOT change the TRAIN/VAL paths in this file (keeps original request).
"""

import os
import re
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ---- Config (do not change paths) ----
TRAIN_CSV = r"data\train.csv"
VAL_CSV   = r"data\val.csv"
MODEL_SAVE_PATH = r"models\protein_classifier.pt"   # file (script will mkdir if needed)

# Strategy choices (tweakable)
USE_ESM = True               # Recommended: compute ESM embeddings and train classifier on them
ESM_PREFERRED = "esm2_t12_35M_UR50D"  # try this (may require more VRAM). fallback to t6 if needed.
EMB_BATCH = 32               # batch size when computing embeddings
AUGMENT_SEQS = True          # apply sequence augmentation when training from raw sequences
MIXUP_ALPHA = 0.3            # mixup strength (used only for embedding branch)
LABEL_SMOOTHING = 0.05       # cross-entropy label smoothing
EPOCHS = 18
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_SEQ_LEN = 512
SEED = 42

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Make sure folders exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)

# -------------------------
# Utilities: augmentation
# -------------------------
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
# mapping: padding=0, unknown=1, real AAs start at 2
AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AA_VOCAB)}
PAD_IDX = 0
UNK_IDX = 1

# groups for BLOSUM-like substitutions (simple biochemical grouping)
AA_GROUPS = [
    list("AVILMFYW"),  # hydrophobic/aromatic
    list("STNQ"),      # polar uncharged
    list("DE"),        # acidic
    list("KRH"),       # basic
    list("GP"),        # special
    list("C")          # cysteine
]
AA_TO_GROUP = {}
for g in AA_GROUPS:
    for a in g:
        AA_TO_GROUP[a] = tuple(g)

def substitute_similar(seq, p=0.08):
    # replace residues (with prob p) by another residue from same group
    seq = list(seq)
    for i, res in enumerate(seq):
        if res not in AA_TO_GROUP:
            continue
        if random.random() < p:
            group = AA_TO_GROUP[res]
            choices = [x for x in group if x != res]
            if choices:
                seq[i] = random.choice(choices)
    return "".join(seq)

def random_mask(seq, p=0.02):
    # mask residues (replace with 'X')
    seq = list(seq)
    for i, _ in enumerate(seq):
        if random.random() < p:
            seq[i] = "X"
    return "".join(seq)

def random_deletion(seq, p=0.02, min_len=30):
    seq = list(seq)
    if len(seq) <= min_len:
        return "".join(seq)
    keep = []
    for ch in seq:
        if random.random() < p:
            continue
        keep.append(ch)
    if len(keep) < min_len:
        # ensure minimum length
        keep = keep[:min_len]
    return "".join(keep)

def kmer_shuffle(seq, k=3, p=0.03):
    if len(seq) < k*2:
        return seq
    if random.random() >= p:
        return seq
    kmers = [seq[i:i+k] for i in range(0, len(seq), k)]
    random.shuffle(kmers)
    return "".join(kmers)[:len(seq)]

def random_crop(seq, max_len=MAX_SEQ_LEN, p=0.05):
    if len(seq) <= max_len or random.random() >= p:
        return seq
    start = random.randint(0, max(0, len(seq) - max_len))
    return seq[start:start+max_len]

def augment_sequence(seq):
    # pipeline of augmentations (probabilities tuned for reasonable diversity)
    seq = substitute_similar(seq, p=0.08)
    seq = random_mask(seq, p=0.03)
    seq = kmer_shuffle(seq, k=3, p=0.02)
    seq = random_deletion(seq, p=0.02)
    seq = random_crop(seq, max_len=MAX_SEQ_LEN, p=0.02)
    return seq

# -------------------------
# Dataset classes
# -------------------------
class SeqDataset(Dataset):
    """Dataset returning integer-encoded sequences (padded) and labels.
       Applies augmentation if augment=True (only used during training)."""
    def __init__(self, csv_file, label_encoder, augment=False):
        self.df = pd.read_csv(csv_file)
        self.label_encoder = label_encoder
        self.augment = augment
        self.seqs = self.df["sequence"].tolist()
        self.labels = label_encoder.transform(self.df["family_code"])

    def encode(self, seq):
        # encode with PAD=0, UNK=1, AA->2..
        enc = [AA_TO_IDX.get(ch, UNK_IDX) for ch in seq]
        if len(enc) > MAX_SEQ_LEN:
            enc = enc[:MAX_SEQ_LEN]
            length = MAX_SEQ_LEN
        else:
            length = len(enc)
            enc += [PAD_IDX] * (MAX_SEQ_LEN - len(enc))
        return np.array(enc, dtype=np.int64), length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self.augment:
            seq = augment_sequence(seq)
        arr, length = self.encode(seq)
        return torch.tensor(arr), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(length, dtype=torch.long)

class EmbeddingDataset(Dataset):
    """Dataset for precomputed embeddings (numpy arrays) + labels"""
    def __init__(self, emb_path, csv_file, label_encoder, mixup=False, mixup_alpha=0.3):
        self.emb = np.load(emb_path)
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.labels = label_encoder.transform(self.df["family_code"])
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        x = self.emb[idx].astype(np.float32)
        y = int(self.labels[idx])
        return x, y

    def collate_fn(self, batch):
        X = np.stack([b[0] for b in batch], axis=0)
        y = np.array([b[1] for b in batch], dtype=np.int64)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

# -------------------------
# ESM Embeddings (optional)
# -------------------------
def compute_esm_embeddings_if_needed(split_name, df, emb_path):
    """
    Compute embeddings and save to emb_path if not present.
    Uses fair-esm; tries preferred model then falls back to smaller model.
    Cleans sequences to prevent tokenizer errors.
    """
    import os, re
    import numpy as np
    import torch
    from tqdm import tqdm

    if os.path.exists(emb_path):
        print(f"Found existing embeddings: {emb_path}")
        return

    try:
        import esm
    except Exception as e:
        raise RuntimeError("Please install fair-esm: pip install fair-esm - for ESM embeddings.") from e

    # choose model function
    preferred = ESM_PREFERRED
    fallback = "esm2_t6_8M_UR50D"
    model = None
    alphabet = None
    model_name_used = None

    for candidate in (preferred, fallback):
        try:
            model_func = getattr(esm.pretrained, candidate)
            model, alphabet = model_func()
            model.eval()
            model_name_used = candidate
            print("Loaded ESM model:", candidate)
            break
        except Exception:
            model = None
            alphabet = None
            continue

    if model is None:
        raise RuntimeError("No suitable ESM model found in fair-esm.pretrained.")

    batch_converter = alphabet.get_batch_converter()

    # 🔹 Clean sequences (UPPERCASE + replace invalid AA with X)
    valid_aas = set("ACDEFGHIKLMNPQRSTVWYX")
    def clean_sequence(seq):
        seq = seq.upper()
        return "".join([aa if aa in valid_aas else "X" for aa in seq])

    sequences = [clean_sequence(s) for s in df["sequence"].tolist()]
    out_embs = []

    # infer repr layer from model name
    m = re.search(r"t(\d+)", model_name_used)
    repr_layer = int(m.group(1)) if m else None

    # compute in batches
    bs = EMB_BATCH if DEVICE.type == "cuda" else max(4, EMB_BATCH // 4)
    with torch.no_grad():
        model = model.to(DEVICE)
        for i in tqdm(range(0, len(sequences), bs), desc=f"ESM embed {split_name}"):
            batch_seqs = sequences[i:i+bs]
            batch = [(str(j), s) for j, s in enumerate(batch_seqs)]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(DEVICE)
            results = model(tokens, repr_layers=[repr_layer] if repr_layer is not None else None)

            if "representations" in results:
                reprs_dict = results["representations"]
                if repr_layer in reprs_dict:
                    reprs = reprs_dict[repr_layer]
                else:
                    reprs = reprs_dict[max(reprs_dict.keys())]
            elif "mean_representations" in results:
                reprs = results["mean_representations"]
            else:
                raise RuntimeError("ESM model returned no representations. API mismatch.")

            # mean pool actual residues (exclude BOS/EOS)
            for j, s in enumerate(batch_seqs):
                L = len(s)
                rep = reprs[j, 1: L+1].mean(0).cpu().numpy()
                out_embs.append(rep)

    out_embs = np.stack(out_embs, axis=0)
    np.save(emb_path, out_embs)
    print("✅ Saved embeddings to", emb_path)

# -------------------------
# Models
# -------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, drop=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvClassifier(nn.Module):
    """1D conv -> global pooling -> FC (used if training from raw sequences)"""
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.conv1 = nn.Conv1d(embed_dim, 256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x)            # (B, L, E)
        x = x.permute(0, 2, 1)           # (B, E, L)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)     # (B, 256)
        return self.fc(x)

# -------------------------
# Training helpers
# -------------------------
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, 1.0, None, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def train_epoch_embeddings(model, loader, optimizer, scaler, criterion, device, mixup_alpha=0.3):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for X, y in tqdm(loader, desc="Train (emb)"):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        if mixup_alpha > 0:
            X_m, y_a, y_b, lam, _ = mixup_data(X, y, alpha=mixup_alpha)
            with torch.amp.autocast(device_type=device.type):
                logits = model(X_m)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            with torch.amp.autocast(device_type=device.type):
                logits = model(X)
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        losses.append(loss.item())
        all_preds.extend(preds)
        all_labels.extend(y.detach().cpu().numpy())
    return np.mean(losses), accuracy_score(all_labels, all_preds)

def eval_embeddings(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Eval (emb)"):
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return accuracy_score(all_labels, all_preds), classification_report(all_labels, all_preds, zero_division=0, output_dict=False)

def train_epoch_sequences(model, loader, optimizer, scaler, criterion, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for seqs, labels, lengths in tqdm(loader, desc="Train (seq)"):
        seqs = seqs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            logits = model(seqs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        losses.append(loss.item())
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
    return np.mean(losses), accuracy_score(all_labels, all_preds)

def eval_sequences(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels, lengths in tqdm(loader, desc="Eval (seq)"):
            seqs = seqs.to(device)
            labels = labels.to(device)
            logits = model(seqs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), classification_report(all_labels, all_preds, zero_division=0, output_dict=False)

# -------------------------
# MAIN: prepare datasets, models, training
# -------------------------
def main():
    # Load dataframes
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print("Train samples:", len(train_df), "| Val samples:", len(val_df))

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["family_code"])

    num_classes = len(label_encoder.classes_)
    print("Num classes:", num_classes)

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    if USE_ESM:
        # compute embeddings if needed
        emb_train_path = "data/embeddings_train.npy"
        emb_val_path   = "data/embeddings_val.npy"

        compute_esm_embeddings_if_needed("train", train_df, emb_train_path)
        compute_esm_embeddings_if_needed("val", val_df, emb_val_path)

        train_ds = EmbeddingDataset(emb_train_path, TRAIN_CSV, label_encoder, mixup=True, mixup_alpha=MIXUP_ALPHA)
        val_ds   = EmbeddingDataset(emb_val_path, VAL_CSV, label_encoder, mixup=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, collate_fn=val_ds.collate_fn)

        # build classifier on top of embeddings
        emb_dim = np.load(emb_train_path).shape[1]
        print("Embedding dim:", emb_dim)
        model = MLPClassifier(input_dim=emb_dim, hidden_dims=[1024, 512], num_classes=num_classes, drop=0.5).to(DEVICE)

        # class weights + label smoothing
        class_counts = train_df["family_code"].value_counts().sort_index()
        weights = torch.tensor(1.0 / class_counts.values, dtype=torch.float32).to(DEVICE)
        # CrossEntropyLoss accepts weight and label_smoothing in modern PyTorch
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        best_val = 0.0
        patience = 5
        no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss, train_acc = train_epoch_embeddings(model, train_loader, optimizer, scaler, criterion, DEVICE, mixup_alpha=MIXUP_ALPHA)
            val_acc, val_report = eval_embeddings(model, val_loader, DEVICE)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print("Val report (partial):\n", val_report)

            # checkpoint best
            if val_acc > best_val:
                best_val = val_acc
                no_improve = 0
                torch.save({"model_state": model.state_dict(), "label_encoder": label_encoder.classes_}, MODEL_SAVE_PATH)
                print("Saved best model →", MODEL_SAVE_PATH)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("No improvement for", patience, "epochs — early stopping.")
                    break

        print("Best val acc:", best_val)
    else:
        # training from raw sequences with augmentations
        train_ds = SeqDataset(TRAIN_CSV, label_encoder, augment=AUGMENT_SEQS)
        val_ds   = SeqDataset(VAL_CSV, label_encoder, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)

        model = ConvClassifier(vocab_size=len(AA_TO_IDX) + 2, embed_dim=128, num_classes=num_classes).to(DEVICE)

        class_counts = train_df["family_code"].value_counts().sort_index()
        weights = torch.tensor(1.0 / class_counts.values, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        best_val = 0.0
        patience = 5
        no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}")
            train_loss, train_acc = train_epoch_sequences(model, train_loader, optimizer, scaler, criterion, DEVICE)
            val_acc, val_report = eval_sequences(model, val_loader, DEVICE)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print("Val report (partial):\n", val_report)

            if val_acc > best_val:
                best_val = val_acc
                no_improve = 0
                torch.save({"model_state": model.state_dict(), "label_encoder": label_encoder.classes_}, MODEL_SAVE_PATH)
                print("Saved best model →", MODEL_SAVE_PATH)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("No improvement for", patience, "epochs — early stopping.")
                    break

        print("Best val acc:", best_val)

if __name__ == "__main__":
    main()
