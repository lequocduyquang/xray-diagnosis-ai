# Train, evaluate, save model

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from checklist_dataset import ChecklistDataset
from checklist_model import ChecklistMLP

# Load CSV
df = pd.read_csv('checklist_data.csv')
feature_cols = ['fever', 'cough', 'fast_breathing', 'chest_indrawing']
label_col = 'label'

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare Dataset
train_dataset = ChecklistDataset(train_df, feature_cols, label_col)
val_dataset = ChecklistDataset(val_df, feature_cols, label_col, scaler=train_dataset.scaler)

# Save scaler
joblib.dump(train_dataset.scaler, 'scaler.pkl')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Init model
model = ChecklistMLP(input_dim=len(feature_cols))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(20):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}, Val Accuracy: {acc:.4f}")

# Save model
torch.save(model.state_dict(), 'checklist_mlp.pth')
