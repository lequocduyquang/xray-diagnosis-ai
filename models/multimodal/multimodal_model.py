# multimodal_model.py

import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    def __init__(self, image_feat_dim=512, checklist_feat_dim=32, hidden_dim=128, output_dim=3):
        super(MultimodalFusionModel, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(image_feat_dim + checklist_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)  # output_dim = số lượng chẩn đoán (VD: Normal / Pneumonia / Others)
        )

    def forward(self, image_feat, checklist_feat):
        # Nối 2 vector đặc trưng lại
        combined = torch.cat((image_feat, checklist_feat), dim=1)
        return self.fusion(combined)
