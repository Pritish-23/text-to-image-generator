# src/models/pipeline.py
# Full text-to-image pipeline
# Task 6 — Text-to-Image Generator Project

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPModel,
    CLIPProcessor
)
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────
# Prompt Router
# ─────────────────────────────────────────
class PromptRouter:
    """
    Routes text prompts to the most suitable model.

    Routing rules:
    - Shape + detail keywords → Attention GAN
    - Shape keywords          → CGAN
    - Flower keywords         → Stable Diffusion
    - Default                 → Stable Diffusion
    """

    def __init__(self):
        self.flower_keywords = [
            'flower', 'rose', 'tulip', 'sunflower',
            'lily', 'orchid', 'daisy', 'lavender',
            'lotus', 'iris', 'primrose', 'dahlia',
            'poppy', 'hibiscus', 'carnation', 'marigold',
            'petunia', 'daffodil', 'magnolia', 'blossom',
            'petal', 'bloom', 'floral', 'botanical',
            'garden'
        ]
        self.shape_keywords = [
            'circle', 'square', 'triangle', 'shape',
            'geometric', 'round', 'rectangle'
        ]
        self.detail_keywords = [
            'detailed', 'intricate', 'complex',
            'elaborate', 'fine', 'precise', 'attention'
        ]

    def route(self, prompt):
        """
        Route prompt to appropriate model.

        Returns:
            model_key : str — 'stable_diffusion',
                               'cgan', or 'attention_gan'
            reason    : str — explanation
        """
        prompt_lower = prompt.lower()

        if any(kw in prompt_lower
               for kw in self.shape_keywords):
            if any(kw in prompt_lower
                   for kw in self.detail_keywords):
                return (
                    'attention_gan',
                    'Shape + detail keywords detected'
                )
            return 'cgan', 'Shape keywords detected'

        if any(kw in prompt_lower
               for kw in self.flower_keywords):
            return (
                'stable_diffusion',
                'Flower keywords detected'
            )

        return (
            'stable_diffusion',
            'No specific keywords — defaulting to SD'
        )


# ─────────────────────────────────────────
# Text Preprocessor
# ─────────────────────────────────────────
class TextPreprocessor:
    """
    Preprocesses text prompts into CLIP embeddings.

    Input  : text prompt string
    Output : normalized 512-dim embedding vector
    """

    def __init__(self, device):
        self.device = device
        print("Loading CLIP text encoder...")
        self.tokenizer  = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.text_model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)
        self.text_model.eval()
        print("Text preprocessor ready ✓")

    def encode(self, prompt):
        """Encode text prompt to normalized embedding."""
        tokens = self.tokenizer(
            prompt,
            padding    = True,
            truncation = True,
            max_length = 77,
            return_tensors = 'pt'
        )
        input_ids      = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(
            self.device
        )

        with torch.no_grad():
            outputs   = self.text_model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )
            embedding = outputs.pooler_output
            embedding = F.normalize(embedding, dim=-1)

        return embedding


# ─────────────────────────────────────────
# Shape Generator
# ─────────────────────────────────────────
class ShapeGenerator:
    """
    Generates geometric shapes using CGAN
    or Attention GAN.
    """

    def __init__(self, generator, use_attention=False):
        self.generator     = generator
        self.use_attention = use_attention
        self.shape_names   = ['circle', 'square', 'triangle']
        self.generator.eval()

    def generate(self, prompt, num_images=1):
        """Generate shape image from prompt."""
        prompt_lower = prompt.lower()
        label        = 0

        if 'square' in prompt_lower or \
           'rectangle' in prompt_lower:
            label = 1
        elif 'triangle' in prompt_lower:
            label = 2

        device  = next(self.generator.parameters()).device
        labels  = torch.LongTensor(
            [label] * num_images
        ).to(device)
        noise   = torch.randn(num_images, 100).to(device)

        with torch.no_grad():
            images = self.generator(noise, labels)
            images = (images + 1) / 2

        pil_images = []
        for img in images:
            img_np  = img.squeeze().cpu().numpy()
            img_np  = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(
                img_np, mode='L'
            ).convert('RGB')
            img_pil = img_pil.resize(
                (512, 512), Image.NEAREST
            )
            pil_images.append(img_pil)

        return pil_images, self.shape_names[label]


# ─────────────────────────────────────────
# Flower Generator
# ─────────────────────────────────────────
class FlowerGenerator:
    """
    Generates flower images using fine-tuned
    Stable Diffusion with LoRA weights.
    """

    def __init__(self, device, lora_path=None):
        self.device = device
        print("Loading Stable Diffusion...")

        self.pipeline = StableDiffusionPipeline\
            .from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype    = torch.float16,
                safety_checker = None
            ).to(device)

        if lora_path:
            print(f"Loading LoRA from {lora_path}...")
            self.pipeline.unet = PeftModel.from_pretrained(
                self.pipeline.unet, lora_path
            )
            self.pipeline.unet = \
                self.pipeline.unet.merge_and_unload()

        self.pipeline.enable_attention_slicing()
        print("Flower generator ready ✓")

    def generate(
        self,
        prompt,
        num_images      = 1,
        steps           = 30,
        guidance_scale  = 7.5,
        seed            = 42
    ):
        """Generate flower image from prompt."""
        images = []
        for i in range(num_images):
            with torch.no_grad():
                image = self.pipeline(
                    prompt              = prompt,
                    negative_prompt     = "blurry, low quality, distorted, ugly",
                    num_inference_steps = steps,
                    guidance_scale      = guidance_scale,
                    height              = 512,
                    width               = 512,
                    generator           = torch.Generator(
                        self.device
                    ).manual_seed(seed + i)
                ).images[0]
            images.append(image)
        return images


# ─────────────────────────────────────────
# CLIP Scorer
# ─────────────────────────────────────────
class CLIPScorer:
    """
    Evaluates text-image alignment using CLIP score.
    Higher score = better alignment.
    """

    def __init__(self, device):
        self.device = device
        print("Loading CLIP scorer...")
        self.model     = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.model.eval()
        print("CLIP scorer ready ✓")

    def score(self, image, prompt):
        """Compute CLIP score between image and prompt."""
        inputs = self.processor(
            text           = [prompt],
            images         = image,
            return_tensors = 'pt',
            padding        = True
        ).to(self.device)

        with torch.no_grad():
            outputs    = self.model(**inputs)
            clip_score = outputs.logits_per_image.item() / 100

        return round(clip_score, 4)


# ─────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────
class TextToImagePipeline:
    """
    Full text-to-image generation pipeline.

    Connects:
    - Text preprocessing (CLIP)
    - Prompt routing
    - Image generation (SD / CGAN / Attention GAN)
    - CLIP score evaluation

    Usage:
        pipeline = TextToImagePipeline(
            device         = device,
            cgan_generator = generator,
            attn_generator = attn_generator,
            lora_path      = 'path/to/lora'
        )
        result = pipeline.generate("a photo of a rose")
        result['image'].show()
    """

    def __init__(
        self,
        device,
        cgan_generator = None,
        attn_generator = None,
        lora_path      = None
    ):
        self.device = device

        # Initialize components
        self.router            = PromptRouter()
        self.text_preprocessor = TextPreprocessor(device)
        self.clip_scorer       = CLIPScorer(device)
        self.history           = []

        # Shape generators
        if cgan_generator:
            self.shape_generator = ShapeGenerator(
                cgan_generator,
                use_attention=False
            )
        if attn_generator:
            self.attn_shape_gen = ShapeGenerator(
                attn_generator,
                use_attention=True
            )

        # Flower generator
        self.flower_generator = FlowerGenerator(
            device, lora_path
        )

        print("\n=== Pipeline Ready ✓ ===")

    def generate(self, prompt, verbose=True):
        """
        Generate image from text prompt.

        Args:
            prompt  : text description
            verbose : print step-by-step progress

        Returns:
            dict with keys:
                prompt     : input prompt
                image      : PIL image
                model      : model used
                clip_score : CLIP alignment score
                embedding  : CLIP text embedding
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"Prompt: {prompt}")
            print(f"{'='*50}")

        # Step 1: Encode text
        if verbose: print("\nStep 1: Encoding text...")
        embedding = self.text_preprocessor.encode(prompt)

        # Step 2: Route prompt
        if verbose: print("Step 2: Routing prompt...")
        model_key, reason = self.router.route(prompt)
        if verbose: print(f"  → {model_key} ({reason})")

        # Step 3: Generate image
        if verbose: print("Step 3: Generating image...")
        if model_key == 'stable_diffusion':
            images = self.flower_generator.generate(prompt)
            source = 'Fine-tuned Stable Diffusion'
        elif model_key == 'attention_gan':
            images, _ = self.attn_shape_gen.generate(prompt)
            source    = 'Attention GAN'
        else:
            images, _ = self.shape_generator.generate(prompt)
            source    = 'CGAN'

        image = images[0]
        if verbose: print(f"  → Generated by: {source}")

        # Step 4: Score with CLIP
        if verbose: print("Step 4: Computing CLIP score...")
        clip_score = self.clip_scorer.score(image, prompt)
        if verbose: print(f"  → CLIP score: {clip_score}")

        # Step 5: Store result
        result = {
            'prompt'    : prompt,
            'image'     : image,
            'model'     : source,
            'clip_score': clip_score,
            'embedding' : embedding.cpu().numpy()
        }
        self.history.append(result)

        if verbose:
            print(f"\n{'='*50}")
            print(f"Complete ✓")
            print(f"{'='*50}\n")

        return result
