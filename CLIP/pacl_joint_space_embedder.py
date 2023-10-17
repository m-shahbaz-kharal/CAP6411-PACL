import torch
from torch import nn

class PACLJointSpaceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.residual_path = nn.Linear(512, 512)

    def forward(self, image_embedding):
        image_embedding = self.main_path(image_embedding)
        image_embedding = image_embedding + self.residual_path(image_embedding)
        return image_embedding