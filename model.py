import timm
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


class ResNetBaseline(nn.Module):
    def __init__(self, out_dim=28, pi_bias=0.01):
        super(ResNetBaseline, self).__init__()

        self.img_model = models.resnet50(pretrained=True)
        self.img_model.fc = nn.Identity()
        self.ht_model = HashTagNetwork(input_dim=768)
        self.fc = nn.Linear(in_features=4096, out_features=out_dim, bias=True)

        # Bias initialization in Appendix A
        nn.init.constant_(self.fc.bias, -torch.log((torch.tensor(1) - pi_bias) / pi_bias))

    def forward(self, x, embed):
        img_logits = self.img_model(x)
        ht_logits = self.ht_model(embed)
        logits = self.fc(torch.cat((img_logits, ht_logits), dim=-1))
        return logits


class VirtexVisual(nn.Module):
    def __init__(self, out_dim=28, pi_bias=0.01):
        super(VirtexVisual, self).__init__()

        self.model = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)

        # Bias initialization in Appendix A
        nn.init.constant_(self.model.fc.bias, -torch.log((torch.tensor(1) - pi_bias) / pi_bias))

    def forward(self, x):
        logits = self.model(x)
        return logits


class Virtex(nn.Module):
    def __init__(self, out_dim=28, pi_bias=0.01):
        super(Virtex, self).__init__()

        self.img_model = torch.hub.load("kdexd/virtex", "resnet50", pretrained=True)
        self.img_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.img_model.fc = nn.Identity()
        self.ht_model = HashTagNetwork(input_dim=768)
        self.fc = nn.Linear(in_features=4096, out_features=out_dim, bias=True)

        # Bias initialization in Appendix A
        nn.init.constant_(self.fc.bias, -torch.log((torch.tensor(1) - pi_bias) / pi_bias))

    def forward(self, x, embed):
        img_logits = self.img_model(x)
        ht_logits = self.ht_model(embed)
        logits = self.fc(torch.cat((img_logits, ht_logits), dim=-1))
        return logits


class SwinTransformerVisual(nn.Module):
    def __init__(self, out_dim=28, model_size='tiny'):
        super(SwinTransformerVisual, self).__init__()

        if model_size == 'tiny':
            self.model = timm.models.swin_transformer.swin_tiny_patch4_window7_224(pretrained=True)
            self.model.head = nn.Linear(in_features=768, out_features=out_dim, bias=True)
        elif model_size == 'small':
            self.model = timm.models.swin_transformer.swin_small_patch4_window7_224(pretrained=True)
            self.model.head = nn.Linear(in_features=768, out_features=out_dim, bias=True)
        elif model_size == 'base':
            self.model = timm.models.swin_transformer.swin_base_patch4_window7_224_in22k(pretrained=True)
            self.model.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)

        # Init: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L372
        nn.init.zeros_(self.model.head.weight)
        head_bias = 0
        nn.init.constant_(self.model.head.bias, head_bias)

    def forward(self, x):
        logits = self.model(x)
        return logits


class SwinTransformer(nn.Module):
    def __init__(self, out_dim=28, model_size='tiny'):
        super(SwinTransformer, self).__init__()

        self.ht_model = HashTagNetwork(input_dim=768)
        if model_size == 'tiny':
            self.img_model = timm.models.swin_transformer.swin_tiny_patch4_window7_224(pretrained=True)
            self.fc = nn.Linear(in_features=768+2048, out_features=out_dim, bias=True)
        elif model_size == 'small':
            self.img_model = timm.models.swin_transformer.swin_small_patch4_window7_224(pretrained=True)
            self.fc = nn.Linear(in_features=768+2048, out_features=out_dim, bias=True)
        elif model_size == 'base':
            self.img_model = timm.models.swin_transformer.swin_base_patch4_window7_224_in22k(pretrained=True)
            self.fc = nn.Linear(in_features=1024+2048, out_features=out_dim, bias=True)
        self.img_model.head = nn.Identity()

        # Init: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L372
        nn.init.zeros_(self.fc.weight)
        head_bias = 0
        nn.init.constant_(self.fc.bias, head_bias)

    def forward(self, x, embed):
        img_logits = self.img_model(x)
        ht_logits = self.ht_model(embed)
        logits = self.fc(torch.cat((img_logits, ht_logits), dim=-1))
        return logits


class HashTagNetwork(nn.Module):
    def __init__(self, input_dim):
        super(HashTagNetwork, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.25),
        )

    def forward(self, x):
        logits = self.linear(x)
        return logits
