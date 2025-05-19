import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet-b0', freeze_backbone=True, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        self.backbone = EfficientNet.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self, unfreeze_ratio=0.2):
        print("🔓 Unfreezing partial backbone...")
        total_layers = list(self.backbone.named_children())
        unfreeze_count = int(len(total_layers) * unfreeze_ratio)
        for name, module in total_layers[-unfreeze_count:]:
            for param in module.parameters():
                param.requires_grad = True
