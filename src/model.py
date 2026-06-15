import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (N, C, H, W) or (N, C)
        if x.dim() == 4:
            z = x.mean(dim=(2, 3))  # (N, C)
        elif x.dim() == 2:
            z = x
        else:
            # fallback: global mean on last dim
            z = x.mean(dim=1)
        z = self.fc1(z)
        z = F.relu(z)
        z = self.fc2(z)
        z = torch.sigmoid(z)
        if x.dim() == 4:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return x * z


class Frontend3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(Frontend3D, self).__init__()
        # 3D conv kernel 5x7x7
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 7, 7), padding=(2, 3, 3), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # max-pooling with temporal=1, spatial=3x3 (stride spatial 2)
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        # x: (B, T, C, H, W) -> Conv3d expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        # output (B, C', T', H', W') -> convert back to (B, T', C', H', W')
        x = x.permute(0, 2, 1, 3, 4)
        return x


class SE_ResNet18_FrameEncoder(nn.Module):
    def __init__(self, in_channels=64, pretrained=False, dropout=0.5):
        super(SE_ResNet18_FrameEncoder, self).__init__()
        # load torchvision resnet18 using the new 'weights' API when available
        try:
            # torchvision >= 0.13
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        except Exception:
            # fallback for older torchvision versions that use pretrained flag
            resnet = models.resnet18(pretrained=pretrained)
        # adapt first conv to accept in_channels
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = resnet
        self.feat_dim = 512
        # SE blocks for each ResNet layer output (channel sizes: 64,128,256,512)
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, frames):
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        # ensure contiguous before reshaping (fixes view() error when tensor is not contiguous)
        x = frames.contiguous().view(B * T, C, H, W)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.se1(x)
        x = self.backbone.layer2(x)
        x = self.se2(x)
        x = self.backbone.layer3(x)
        x = self.se3(x)
        x = self.backbone.layer4(x)
        x = self.se4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # (B*T, 512)
        # apply dropout per-frame
        x = self.dropout(x)
        x = x.view(B, T, -1)  # (B, T, feat_dim)
        return x


class BGCLBackEnd(nn.Module):
    def __init__(self, input_dim, num_classes, gru_hidden=1024, gru_layers=3, conv_channels=(1024, 512, 512, 256), conv_kernels=(32, 32, 16, 16), dropout=0.5):
        super(BGCLBackEnd, self).__init__()
        # Bi-GRU: bidirectional GRU with multiple layers
        self.gru = nn.GRU(input_dim, gru_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.post_conv = nn.ModuleList()
        in_ch = gru_hidden * 2
        for out_ch, k in zip(conv_channels, conv_kernels):
            self.post_conv.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        self.dropout = nn.Dropout(p=dropout)
        # fully connected maps per timestep features to class logits
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, D)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.gru(x)
        # out: (B, T, 2*gru_hidden)
        out = out.permute(0, 2, 1)  # (B, C, T)
        for layer in self.post_conv:
            out = layer(out)
        # out: (B, C, T)
        out = out.permute(0, 2, 1)  # (B, T, C)
        out = self.dropout(out)
        logits_t = self.fc(out)  # (B, T, num_classes)
        return logits_t


class FullModel(nn.Module):
    def __init__(self, num_classes, in_channels=1, frontend_out=64, pretrained_resnet=False, dropout=0.5):
        super(FullModel, self).__init__()
        self.frontend = Frontend3D(in_channels=in_channels, out_channels=frontend_out)
        self.encoder = SE_ResNet18_FrameEncoder(in_channels=frontend_out, pretrained=pretrained_resnet, dropout=dropout)
        self.backend = BGCLBackEnd(input_dim=self.encoder.feat_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x, lengths=None):
        # x: (B, T, C, H, W)
        x = x.float()
        x = self.frontend(x)  # (B, T', C', H', W')
        features = self.encoder(x)  # (B, T', feat_dim)
        logits_t = self.backend(features, lengths=lengths)  # (B, T', num_classes)
        # temporal GAP (masked)
        if lengths is None:
            logits = logits_t.mean(dim=1)
        else:
            B, T, C = logits_t.shape
            device = logits_t.device
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)
            summed = (logits_t * mask).sum(dim=1)
            denom = lengths.to(device).unsqueeze(1).clamp(min=1).float()
            logits = summed / denom
        return logits, logits_t