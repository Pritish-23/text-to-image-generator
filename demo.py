# demo.py
# Run this file to launch the Gradio UI locally
# For Colab: run the notebook instead

import torch
import gradio as gr
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel

# Configuration

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_ID   = "runwayml/stable-diffusion-v1-5"
LORA_PATH  = "sd_output/lora_weights"

print(f"Loading pipeline on {DEVICE}...")

# Load pipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype    = torch.float16,
    safety_checker = None
).to(DEVICE)

pipeline.unet = PeftModel.from_pretrained(
    pipeline.unet, LORA_PATH
)
pipeline.unet = pipeline.unet.merge_and_unload()
pipeline.enable_attention_slicing()

print("Pipeline ready ✓")


# Generation function

def generate(prompt, steps, guidance, seed):
    with torch.no_grad():
        image = pipeline(
            prompt              = prompt,
            negative_prompt     = "blurry, low quality, distorted",
            num_inference_steps = steps,
            guidance_scale      = guidance,
            height              = 512,
            width               = 512,
            generator           = torch.Generator(DEVICE).manual_seed(seed)
        ).images[0]
    return image


# Gradio UI

with gr.Blocks(title="Text-to-Image Generator") as demo:
    gr.Markdown("# 🌸 Real-Time Text-to-Image Generator")

    with gr.Row():
        with gr.Column():
            prompt  = gr.Textbox(label="Prompt", lines=3)
            steps   = gr.Slider(10, 50, value=30, label="Steps")
            guidance= gr.Slider(1, 15, value=7.5, label="Guidance")
            seed    = gr.Slider(0, 1000, value=42, label="Seed")
            btn     = gr.Button("✨ Generate", variant="primary")

        with gr.Column():
            output  = gr.Image(label="Generated Image")

    btn.click(
        fn      = generate,
        inputs  = [prompt, steps, guidance, seed],
        outputs = output
    )

if __name__ == "__main__":
    demo.launch()
