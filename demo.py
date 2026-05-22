# demo.py
# Run this file to launch the Gradio UI locally
#
# Usage:
#   pip install -r requirements.txt
#   python demo.py
#
# Requirements:
#   - LoRA weights saved at sd_output/lora_weights/
#   - GPU recommended for faster generation

import torch
import gradio as gr
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_ID    = "runwayml/stable-diffusion-v1-5"
LORA_PATH   = "sd_output/lora_weights"

print(f"Device     : {DEVICE}")
print(f"LoRA path  : {LORA_PATH}")

# ─────────────────────────────────────────
# Load pipeline
# ─────────────────────────────────────────
print("\nLoading Stable Diffusion...")

pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype    = torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker = None
).to(DEVICE)

# Load LoRA weights if available
if os.path.exists(LORA_PATH):
    print("Loading LoRA weights...")
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet, LORA_PATH
    )
    pipeline.unet = pipeline.unet.merge_and_unload()
    print("LoRA weights loaded ✓")
else:
    print("LoRA weights not found — using base SD")

pipeline.enable_attention_slicing()
print("Pipeline ready ✓")

# ─────────────────────────────────────────
# Generation function
# ─────────────────────────────────────────
def generate(prompt, steps, guidance, seed):
    if not prompt.strip():
        return None, "Please enter a prompt."

    try:
        with torch.no_grad():
            image = pipeline(
                prompt              = prompt,
                negative_prompt     = "blurry, low quality, distorted, ugly",
                num_inference_steps = int(steps),
                guidance_scale      = float(guidance),
                height              = 512,
                width               = 512,
                generator           = torch.Generator(DEVICE).manual_seed(int(seed))
            ).images[0]

        info = f"""Generation complete ✓
─────────────────────────
Prompt     : {prompt[:50]}
Steps      : {steps}
Guidance   : {guidance}
Seed       : {seed}
Device     : {DEVICE}
─────────────────────────"""

        return image, info

    except Exception as e:
        blank = Image.fromarray(
            np.zeros((512, 512, 3), dtype=np.uint8)
        )
        return blank, f"Error: {str(e)}"

# ─────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────
with gr.Blocks(title="🌸 Text-to-Image Generator") as demo:

    gr.Markdown("""
    # 🌸 Real-Time Text-to-Image Generator
    ### Fine-tuned Stable Diffusion on Oxford-102 Flowers
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ✍️ Input")

            prompt_input = gr.Textbox(
                label       = "Text Prompt",
                placeholder = "a photo of a sunflower, vibrant colors...",
                lines       = 3
            )

            with gr.Row():
                steps = gr.Slider(
                    label   = "Inference Steps",
                    minimum = 10,
                    maximum = 50,
                    value   = 30,
                    step    = 1
                )
                guidance = gr.Slider(
                    label   = "Guidance Scale",
                    minimum = 1.0,
                    maximum = 15.0,
                    value   = 7.5,
                    step    = 0.5
                )

            seed = gr.Slider(
                label   = "Seed",
                minimum = 0,
                maximum = 1000,
                value   = 42,
                step    = 1
            )

            btn = gr.Button(
                "✨ Generate Image",
                variant = "primary"
            )

            gr.Markdown("### 💡 Example Prompts")
            gr.Examples(
                examples = [
                    ["a photo of a sunflower, vibrant colors, nature photography",          30, 7.5, 42],
                    ["a photo of a rose flower, highly detailed, vibrant colors",            30, 7.5, 43],
                    ["a photo of a lotus flower, nature photography",                        30, 7.5, 44],
                    ["a photo of a hibiscus flower, vibrant colors",                         30, 7.5, 45],
                    ["a photo of a lavender flower, vibrant colors",                         30, 9.0, 46],
                    ["a photo of a purple coneflower, nature photography",                   30, 7.5, 47],
                    ["a photo of a pink primrose flower, detailed petals",                   30, 7.5, 48],
                    ["a photo of a tiger lily flower, vibrant colors, nature photography",   30, 7.5, 49],
                ],
                inputs = [prompt_input, steps, guidance, seed]
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Generated Image")

            image_output = gr.Image(
                label  = "Generated Image",
                height = 512,
                width  = 512
            )

            info_output = gr.Textbox(
                label       = "Generation Info",
                lines       = 8,
                interactive = False
            )

    gr.Markdown("""
    ---
    **Model:** Stable Diffusion v1.5 (LoRA fine-tuned) |
    **Dataset:** Oxford-102 Flowers |
    **Built with:** PyTorch · Diffusers · PEFT · Gradio
    """)

    btn.click(
        fn      = generate,
        inputs  = [prompt_input, steps, guidance, seed],
        outputs = [image_output, info_output]
    )

if __name__ == "__main__":
    demo.launch()
