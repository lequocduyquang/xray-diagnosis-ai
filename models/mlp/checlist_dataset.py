# Dataset _ Scaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

class ChecklistDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, scaler=None):
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(df[feature_cols].values)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(df[feature_cols].values)
        self.y = df[label_col].values.astype('int64')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).long()
