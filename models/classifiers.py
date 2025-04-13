import torch
import torch.nn as nn

class HierarchicalPartialClassifier(nn.Module):
    def __init__(self, in_features=2048, num_fine_classes=15, num_coarse_classes=5):
        super().__init__()
        self.coarse_classifier = nn.Linear(in_features, num_coarse_classes)
        self.fine_classifier = nn.Linear(in_features, num_fine_classes)
        self.threshold = nn.Parameter(torch.tensor(0.5))  # Learnable threshold
    def forward(self, x):
        coarse_logits = self.coarse_classifier(x)
        fine_logits = self.fine_classifier(x)
        max_coarse_prob = torch.softmax(coarse_logits, dim=1).max(dim=1)[0]
        mask = (max_coarse_prob < self.threshold).unsqueeze(1)
        return torch.where(mask, coarse_logits, fine_logits)