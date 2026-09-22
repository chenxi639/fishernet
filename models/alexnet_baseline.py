"""Whole-image AlexNet baseline before SPP and Fisher integration."""
import torch
from torch import nn
from torchvision.models import alexnet


class AlexNetBaseline(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        base = alexnet(weights=None)
        if pretrained_path is not None:
            base.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        self.features = base.features
        self.avgpool = base.avgpool
        self.fc67 = nn.Sequential(*list(base.classifier.children())[:-1])
        self.embedding = nn.Linear(4096, 256)
        self.head = nn.Linear(256, 20)
        for layer in (self.embedding, self.head):
            nn.init.normal_(layer.weight, mean=0., std=.01)
            nn.init.zeros_(layer.bias)

    def forward_features(self, x):
        x = self.avgpool(self.features(x)).flatten(1)
        return self.embedding(self.fc67(x))

    def forward(self, x):
        return self.head(self.forward_features(x))

    def optimizer_groups(self, backbone_lr=.001, new_lr=.01):
        return [dict(params=list(self.features.parameters())+list(self.fc67.parameters()), lr=backbone_lr),
                dict(params=list(self.embedding.parameters())+list(self.head.parameters()), lr=new_lr)]
