import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from checklist_dataset import ChecklistDataset
from checklist_model import ChecklistMLP

# Đọc dữ liệu từ CSV
df = pd.read_csv('checklist_data.csv')

# Cấu hình các cột dữ liệu
feature_cols = ['fever', 'cough', 'fast_breathing', 'chest_indrawing']
disease_col = 'disease'
symptoms_cols = ['Fever', 'Cough', 'Fast Breathing', 'Chest Indrawing']
recommendation_col = 'recommendation'

# Chia dữ liệu thành các nhãn
df['label_disease'] = df[disease_col].map({'Pneumonia': 0, 'Flu': 1, 'Normal': 2, 'Cold': 3})
df['label_symptoms'] = df[symptoms_cols].apply(lambda x: [1 if symptom in x else 0 for symptom in symptoms_cols], axis=1)
df['label_recommendation'] = df[recommendation_col].apply(lambda x: 1 if 'doctor' in x else 0)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tạo dataset
train_dataset = ChecklistDataset(train_df, feature_cols, disease_col, symptoms_cols, recommendation_col, scaler=None)
val_dataset = ChecklistDataset(val_df, feature_cols, disease_col, symptoms_cols, recommendation_col, scaler=train_dataset.scaler)

models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

os.makedirs(models_dir, exist_ok=True)

# Lưu scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
joblib.dump(train_dataset.scaler, scaler_path)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Khởi tạo mô hình
model = ChecklistMLP(input_dim=len(feature_cols), num_diseases=4, num_symptoms=4)
criterion_disease = nn.CrossEntropyLoss()
criterion_symptoms = nn.BCEWithLogitsLoss()
criterion_recommendation = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
for epoch in range(20):
    model.train()
    for x, y_disease, y_symptoms, y_recommendation in train_loader:
        optimizer.zero_grad()
        disease_logits, symptoms_logits, recommendation_logits = model(x)
        
        disease_loss = criterion_disease(disease_logits, y_disease)
        symptoms_loss = criterion_symptoms(symptoms_logits, y_symptoms)
        recommendation_loss = criterion_recommendation(recommendation_logits, y_recommendation)
        
        total_loss = disease_loss + symptoms_loss + recommendation_loss
        total_loss.backward()
        optimizer.step()

    # Đánh giá mô hình
    model.eval()
    all_preds_disease, all_labels_disease = [], []
    all_preds_symptoms, all_labels_symptoms = [], []
    all_preds_recommendation, all_labels_recommendation = [], []
    with torch.no_grad():
        for x, y_disease, y_symptoms, y_recommendation in val_loader:
            disease_logits, symptoms_logits, recommendation_logits = model(x)
            
            disease_pred = torch.argmax(disease_logits, dim=1)
            symptoms_pred = (torch.sigmoid(symptoms_logits) > 0.5).int()
            recommendation_pred = (torch.sigmoid(recommendation_logits) > 0.5).int()
            
            all_preds_disease.extend(disease_pred.tolist())
            all_labels_disease.extend(y_disease.tolist())
            
            all_preds_symptoms.extend(symptoms_pred.tolist())
            all_labels_symptoms.extend(y_symptoms.tolist())
            
            all_preds_recommendation.extend(recommendation_pred.tolist())
            all_labels_recommendation.extend(y_recommendation.tolist())
    
    disease_acc = accuracy_score(all_labels_disease, all_preds_disease)
    symptoms_acc = accuracy_score(all_labels_symptoms, all_preds_symptoms)
    recommendation_acc = accuracy_score(all_labels_recommendation, all_preds_recommendation)
    
    print(f"Epoch {epoch+1}, Disease Accuracy: {disease_acc:.4f}, Symptoms Accuracy: {symptoms_acc:.4f}, Recommendation Accuracy: {recommendation_acc:.4f}")

torch.save(model.state_dict(), os.path.join(models_dir, 'checklist_mlp.pth'))
