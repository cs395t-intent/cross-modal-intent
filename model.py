import torch
import torchvision.models as models
import torch.nn as nn


class ResNetVisualBaseline(nn.Module):
    def __init__(self, out_dim=28, pi_bias=0.01):
        super(ResNetVisualBaseline, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)

        # Bias initialization in Appendix A
        nn.init.constant_(self.model.fc.bias, -torch.log((torch.tensor(1) - pi_bias) / pi_bias))

    def forward(self, x):
        logits = self.model(x)
        return logits


class HashTagNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(2048,28)
        )

    def forward(self, x):
        logits = self.linear(x)
        return logits
