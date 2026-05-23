# src/utils.py
# Utility functions for the project

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from PIL import Image


def get_device():
    """Returns available device (GPU or CPU)."""
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    return device


def normalize_embeddings(embeddings):
    """L2 normalize embedding vectors."""
    return normalize(embeddings)


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor to displayable numpy array.
    Input  : tensor [C, H, W] normalized to [-1, 1]
    Output : numpy array [H, W, C] in range [0, 1]
    """
    image = tensor.permute(1, 2, 0).numpy()
    image = (image * 0.5) + 0.5
    return np.clip(image, 0, 1)


def plot_loss_curves(
    g_losses,
    d_losses,
    title='Training Losses',
    save_path=None
):
    """Plot generator and discriminator loss curves."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator',
             color='#E74C3C', linewidth=1.5)
    plt.plot(d_losses, label='Discriminator',
             color='#3498DB', linewidth=1.5)
    plt.title(title, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(g_losses, label='Generator',
             color='#E74C3C', linewidth=1.5)
    plt.plot(d_losses, label='Discriminator',
             color='#3498DB', linewidth=1.5)
    plt.title(f'{title} (Log Scale)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def save_image_grid(
    images,
    labels,
    class_names,
    title='Generated Images',
    save_path=None
):
    """Save a grid of generated images."""
    n    = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3, rows * 3))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for ax, img, label in zip(axes, images, labels):
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(class_names[label], fontsize=8)
        ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def compute_clip_score(
    image,
    prompt,
    clip_model,
    clip_processor,
    device
):
    """
    Compute CLIP score between image and prompt.
    Higher score = better text-image alignment.
    """
    inputs = clip_processor(
        text          = [prompt],
        images        = image,
        return_tensors = 'pt',
        padding       = True
    ).to(device)

    with torch.no_grad():
        outputs    = clip_model(**inputs)
        clip_score = outputs.logits_per_image.item() / 100

    return round(clip_score, 4)


def print_model_summary(model, model_name='Model'):
    """Print model parameter count."""
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"\n=== {model_name} Summary ===")
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {total - trainable:,}")
    print(f"Trainable %          : {trainable/total*100:.2f}%")
