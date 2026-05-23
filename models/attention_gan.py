# src/models/attention_gan.py
# Attention-enhanced GAN implementation
# Task 5 — Text-to-Image Generator Project

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────
LATENT_DIM  = 100
NUM_CLASSES = 3
EMBED_DIM   = 50
IMG_SIZE    = 64
CHANNELS    = 1


# ─────────────────────────────────────────
# Self-Attention Module
# ─────────────────────────────────────────
class SelfAttention(nn.Module):
    """
    Self-Attention module.
    Every position in the feature map attends
    to every other position.

    Input  : [batch, channels, H, W]
    Output : [batch, channels, H, W] — same shape
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query   = nn.Conv2d(
            in_channels, in_channels // 8,
            kernel_size=1
        )
        self.key     = nn.Conv2d(
            in_channels, in_channels // 8,
            kernel_size=1
        )
        self.value   = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=1
        )
        self.gamma   = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Query, Key, Value projections
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W)

        # Attention map
        attention = self.softmax(torch.bmm(Q, K))

        # Apply attention to value
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        # Residual connection
        return self.gamma * out + x


# ─────────────────────────────────────────
# Cross-Attention Module
# ─────────────────────────────────────────
class CrossAttention(nn.Module):
    """
    Cross-Attention module.
    Image features attend to label embedding —
    keeps generation true to requested label.

    Input  : x [batch, channels, H, W]
             label_embed [batch, embed_dim]
    Output : [batch, channels, H, W] — same shape
    """

    def __init__(self, in_channels, embed_dim):
        super(CrossAttention, self).__init__()

        self.in_channels = in_channels
        self.query       = nn.Linear(
            embed_dim, in_channels // 8
        )
        self.key         = nn.Conv2d(
            in_channels, in_channels // 8,
            kernel_size=1
        )
        self.value       = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=1
        )
        self.gamma       = nn.Parameter(torch.zeros(1))
        self.softmax     = nn.Softmax(dim=-1)

    def forward(self, x, label_embed):
        B, C, H, W = x.shape

        # Query from label → [B, 1, C//8]
        Q = self.query(label_embed).unsqueeze(1)

        # Key and Value from image
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W)

        # Attention between label and image
        attention = self.softmax(torch.bmm(Q, K))

        # Apply attention
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(B, C, 1, 1).expand(B, C, H, W)

        return self.gamma * out + x


# ─────────────────────────────────────────
# Attention Generator
# ─────────────────────────────────────────
class AttentionGenerator(nn.Module):
    """
    Attention-enhanced Generator.
    Adds self-attention at 16×16 and
    cross-attention at 32×32 resolution.

    Input:
        noise  : [batch, LATENT_DIM]
        labels : [batch] — class indices

    Output:
        image  : [batch, CHANNELS, IMG_SIZE, IMG_SIZE]
    """

    def __init__(self):
        super(AttentionGenerator, self).__init__()

        # Label embedding
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)

        # Initial linear layer
        self.linear = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True)
        )

        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        # Self-attention at 16×16
        self.self_attn = SelfAttention(64)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # Cross-attention at 32×32
        self.cross_attn = CrossAttention(32, EMBED_DIM)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(
                32, CHANNELS,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels)
        x           = torch.cat([noise, label_embed], dim=1)
        x           = self.linear(x)
        x           = x.view(x.size(0), 256, 4, 4)
        x           = self.up1(x)
        x           = self.up2(x)
        x           = self.self_attn(x)
        x           = self.up3(x)
        x           = self.cross_attn(x, label_embed)
        x           = self.up4(x)
        return x


# ─────────────────────────────────────────
# Attention Discriminator
# ─────────────────────────────────────────
class AttentionDiscriminator(nn.Module):
    """
    Attention-enhanced Discriminator.
    Adds self-attention at 16×16 and
    cross-attention at 8×8 resolution.

    Input:
        image  : [batch, CHANNELS, IMG_SIZE, IMG_SIZE]
        labels : [batch] — class indices

    Output:
        score  : [batch, 1] — probability of being real
    """

    def __init__(self):
        super(AttentionDiscriminator, self).__init__()

        # Label embedding
        self.label_emb = nn.Embedding(
            NUM_CLASSES, IMG_SIZE * IMG_SIZE
        )

        # Downsample blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(
                2, 32,
                kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(
                32, 64,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Self-attention at 16×16
        self.self_attn = SelfAttention(64)

        self.down3 = nn.Sequential(
            nn.Conv2d(
                64, 128,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Cross-attention at 8×8
        self.cross_attn = CrossAttention(128, EMBED_DIM)

        self.down4 = nn.Sequential(
            nn.Conv2d(
                128, 256,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image, labels):
        label_embed = self.label_emb(labels)
        label_map   = label_embed.view(
            labels.size(0), 1, IMG_SIZE, IMG_SIZE
        )
        x         = torch.cat([image, label_map], dim=1)
        x         = self.down1(x)
        x         = self.down2(x)
        x         = self.self_attn(x)
        x         = self.down3(x)
        raw_embed = self.label_emb(labels)[:, :EMBED_DIM]
        x         = self.cross_attn(x, raw_embed)
        x         = self.down4(x)
        return self.classifier(x)


# ─────────────────────────────────────────
# Load pretrained Attention GAN
# ─────────────────────────────────────────
def load_attention_gan(
    generator_path,
    discriminator_path,
    device
):
    """
    Load pretrained Attention GAN weights.

    Args:
        generator_path     : path to generator.pth
        discriminator_path : path to discriminator.pth
        device             : torch device

    Returns:
        generator, discriminator
    """
    generator     = AttentionGenerator().to(device)
    discriminator = AttentionDiscriminator().to(device)

    generator.load_state_dict(
        torch.load(generator_path, map_location=device)
    )
    discriminator.load_state_dict(
        torch.load(discriminator_path, map_location=device)
    )

    generator.eval()
    discriminator.eval()

    print("Attention GAN loaded successfully ✓")
    return generator, discriminator
