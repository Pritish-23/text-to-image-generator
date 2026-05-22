# src/utils.py
# Utility functions for the project

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def get_device():
    """Returns available device."""
    return torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )


def normalize_embeddings(embeddings):
    """L2 normalize embedding vectors."""
    return normalize(embeddings)


def tensor_to_image(tensor):
    """Convert PyTorch tensor to displayable numpy array."""
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    return np.clip(image, 0, 1)


def plot_loss_curves(g_losses, d_losses, title='Training Losses'):
    """Plot generator and discriminator loss curves."""
    plt.figure(figsize=(12, 4))
    plt.plot(g_losses, label='Generator',     color='#E74C3C')
    plt.plot(d_losses, label='Discriminator', color='#3498DB')
    plt.title(title, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_generated_images(images, labels, shape_names, path):
    """Save a grid of generated images."""
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            images[i].squeeze().cpu().numpy(),
            cmap='gray'
        )
        ax.set_title(shape_names[labels[i].item()],
                     fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
