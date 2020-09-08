import torch
import torch.nn as nn
from torch.cuda import amp
from efficientnet_pytorch import EfficientNet

class EmbeddigsNet(nn.Module):
    def __init__(self, model_name, embeddings_size, DP=False):
        super(EmbeddigsNet, self).__init__()
        self.effnet = EfficientNet.from_name(model_name)
        self.DP = DP
        # Unfreeze model weights
        for param in self.effnet.parameters():
            param.requires_grad = True
        num_ftrs = self.effnet._fc.in_features
        self.effnet._fc = nn.Linear(num_ftrs, embeddings_size)

    def forward(self, x):
        if self.DP:
            with amp.autocast():
                x = self.effnet(x)
        else:
            x = self.effnet(x)
        return x


def get_model(model_name, embeddings_size, DP=False):
    model = EmbeddigsNet(model_name, embeddings_size, DP)
    return model