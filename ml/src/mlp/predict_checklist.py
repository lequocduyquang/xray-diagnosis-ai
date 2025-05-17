# Load model & predict (for API or CLI)

import torch
import joblib
from checklist_model import ChecklistMLP

# Load model
model = ChecklistMLP(input_dim=4)
model.load_state_dict(torch.load('../models/checklist_mlp.pth'))
model.eval()

# Load scaler
scaler = joblib.load('scaler.pkl')

# Predict function
def predict_from_checklist(checklist_input):
    x = scaler.transform([checklist_input])
    x_tensor = torch.tensor(x).float()
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = 'Normal' if pred == 0 else 'Pneumonia'
        return label, probs[0].tolist()
