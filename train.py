# =============================================================================
# SoundSense — CNN Baseline Training avec détection d'overfitting
# Input  : data/processed/*.npy (mel-spectrograms)
# Output : models/cnn_baseline.pth + courbes loss/AUC
# =============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_PATH = r"C:\soundsense\data\processed"
MODEL_PATH     = r"C:\soundsense\models"
MACHINE_IDS    = ["id_00", "id_02", "id_04", "id_06"]
EPOCHS         = 20
BATCH_SIZE     = 32
LR             = 0.001

os.makedirs(MODEL_PATH, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
class SoundDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
X_all, y_all = [], []

for machine_id in MACHINE_IDS:
    normal   = np.load(os.path.join(PROCESSED_PATH, f"{machine_id}_normal.npy"))
    abnormal = np.load(os.path.join(PROCESSED_PATH, f"{machine_id}_abnormal.npy"))
    X_all.append(normal)
    X_all.append(abnormal)
    y_all.append(np.zeros(len(normal)))
    y_all.append(np.ones(len(abnormal)))

X = np.concatenate(X_all)
y = np.concatenate(y_all)
print(f"Total samples: {len(X)} | Normal: {int((y==0).sum())} | Abnormal: {int((y==1).sum())}")

# ── Train/Val/Test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_loader = DataLoader(SoundDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SoundDataset(X_val, y_val),     batch_size=BATCH_SIZE)
test_loader  = DataLoader(SoundDataset(X_test, y_test),   batch_size=BATCH_SIZE)

# ── CNN Model ──────────────────────────────────────────────────────────────────
class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze()

# ── Training ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model     = CNNBaseline().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Tracking des métriques
train_losses, val_losses, val_aucs, train_aucs = [], [], [], []

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    train_loss = 0
    train_preds_ep, train_labels_ep = [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_preds_ep.extend(preds.detach().cpu().numpy())
        train_labels_ep.extend(y_batch.cpu().numpy())

    # ── Validation ──
    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds    = model(X_batch)
            loss     = criterion(preds, y_batch)
            val_loss += loss.item()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    # ── Métriques ──
    t_loss = train_loss / len(train_loader)
    v_loss = val_loss   / len(val_loader)
    t_auc  = roc_auc_score(train_labels_ep, train_preds_ep)
    v_auc  = roc_auc_score(val_labels, val_preds)

    train_losses.append(t_loss)
    val_losses.append(v_loss)
    train_aucs.append(t_auc)
    val_aucs.append(v_auc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Train AUC: {t_auc:.4f} | Val AUC: {v_auc:.4f}")

# ── Courbes overfitting ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses,   label="Val Loss")
axes[0].set_title("Loss curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(train_aucs, label="Train AUC")
axes[1].plot(val_aucs,   label="Val AUC")
axes[1].set_title("AUC-ROC curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "training_curves.png"))
plt.show()
print("✅ Courbes sauvegardées dans models/training_curves.png")

# ── Evaluation finale ──────────────────────────────────────────────────────────
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds   = model(X_batch).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(y_batch.numpy())

test_preds_binary = [1 if p > 0.5 else 0 for p in test_preds]
print("\n=== TEST RESULTS ===")
print(f"AUC-ROC: {roc_auc_score(test_labels, test_preds):.4f}")
print(classification_report(test_labels, test_preds_binary, target_names=["Normal", "Abnormal"]))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, test_preds_binary))

# ── Save model ─────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(MODEL_PATH, "cnn_baseline.pth"))
print(f"\n✅ Model saved to {MODEL_PATH}/cnn_baseline.pth")