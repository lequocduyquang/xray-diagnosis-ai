import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

class ChecklistDataset(Dataset):
    def __init__(self, df, feature_cols, disease_col, symptoms_cols, recommendation_col, scaler=None):
        # Tạo scaler nếu chưa có và chuẩn hóa dữ liệu
        if scaler is None:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(df[feature_cols].values)
        else:
            self.scaler = scaler
            self.X = self.scaler.transform(df[feature_cols].values)
        
        # Dữ liệu nhãn
        self.y_disease = df[disease_col].values
        self.y_symptoms = df[symptoms_cols].apply(lambda x: eval(x)).values
        self.y_recommendation = df[recommendation_col].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).float(), torch.tensor(self.y_disease[idx]).long(), \
               torch.tensor(self.y_symptoms[idx]).float(), self.y_recommendation[idx]
