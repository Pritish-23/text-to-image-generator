# src/models/cgan.py
# Conditional GAN implementation from scratch
# Task 2 — Text-to-Image Generator Project

import torch
import torch.nn as nn

# ─────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────
LATENT_DIM  = 100
NUM_CLASSES = 3
EMBED_DIM   = 50
IMG_SIZE    = 64
CHANNELS    = 1


# ─────────────────────────────────────────
# Generator
# ─────────────────────────────────────────
class Generator(nn.Module):
    """
    Conditional GAN Generator.
    Takes random noise + class label → generates image.

    Input:
        noise  : [batch, LATENT_DIM]
        labels : [batch] — class indices (0, 1, 2)

    Output:
        image  : [batch, CHANNELS, IMG_SIZE, IMG_SIZE]
    """

    def __init__(self):
        super(Generator, self).__init__()

        # Label embedding — converts label to vector
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)

        # Main network
        self.model = nn.Sequential(

            # Layer 1: 150 → 256 × 4 × 4
            nn.Linear(LATENT_DIM + EMBED_DIM, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),

            # Layer 2: 4×4 → 8×8
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 3: 8×8 → 16×16
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 4: 16×16 → 32×32
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Layer 5: 32×32 → 64×64
            nn.ConvTranspose2d(
                32, CHANNELS,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed label
        label_embedding = self.label_emb(labels)

        # Combine noise and label
        x = torch.cat([noise, label_embedding], dim=1)

        # Linear layer + reshape
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = x.view(x.size(0), 256, 4, 4)

        # Convolutional layers
        x = self.model[3:](x)
        return x


# ─────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Conditional GAN Discriminator.
    Takes image + class label → real or fake score.

    Input:
        image  : [batch, CHANNELS, IMG_SIZE, IMG_SIZE]
        labels : [batch] — class indices (0, 1, 2)

    Output:
        score  : [batch, 1] — probability of being real
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Label embedding — converts label to image size
        self.label_emb = nn.Embedding(
            NUM_CLASSES, IMG_SIZE * IMG_SIZE
        )

        # Main network
        self.model = nn.Sequential(

            # Layer 1: 64×64 → 32×32
            nn.Conv2d(
                2, 32,
                kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32×32 → 16×16
            nn.Conv2d(
                32, 64,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 16×16 → 8×8
            nn.Conv2d(
                64, 128,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 8×8 → 4×4
            nn.Conv2d(
                128, 256,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Classifier
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image, labels):
        # Embed label and reshape to image size
        label_embedding = self.label_emb(labels)
        label_map       = label_embedding.view(
            labels.size(0), 1, IMG_SIZE, IMG_SIZE
        )

        # Concatenate image and label
        x = torch.cat([image, label_map], dim=1)
        return self.model(x)


# ─────────────────────────────────────────
# Load pretrained CGAN
# ─────────────────────────────────────────
def load_cgan(generator_path, discriminator_path, device):
    """
    Load pretrained CGAN weights.

    Args:
        generator_path     : path to generator.pth
        discriminator_path : path to discriminator.pth
        device             : torch device

    Returns:
        generator, discriminator
    """
    generator     = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator.load_state_dict(
        torch.load(generator_path, map_location=device)
    )
    discriminator.load_state_dict(
        torch.load(discriminator_path, map_location=device)
    )

    generator.eval()
    discriminator.eval()

    print("CGAN loaded successfully ✓")
    return generator, discriminator


# ─────────────────────────────────────────
# Generate shapes
# ─────────────────────────────────────────
def generate_shapes(generator, labels, device, latent_dim=100):
    """
    Generate shape images using trained generator.

    Args:
        generator  : trained Generator model
        labels     : list of class indices [0=circle, 1=square, 2=triangle]
        device     : torch device
        latent_dim : size of noise vector

    Returns:
        images : list of PIL images
    """
    from PIL import Image
    import numpy as np

    generator.eval()
    labels_tensor = torch.LongTensor(labels).to(device)
    noise         = torch.randn(
        len(labels), latent_dim
    ).to(device)

    with torch.no_grad():
        images = generator(noise, labels_tensor)
        images = (images + 1) / 2

    pil_images = []
    for img in images:
        img_np  = img.squeeze().cpu().numpy()
        img_np  = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, mode='L').convert('RGB')
        img_pil = img_pil.resize((512, 512), Image.NEAREST)
        pil_images.append(img_pil)

    return pil_images
