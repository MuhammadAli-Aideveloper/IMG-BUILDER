import os
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# 1. Setup Environment and Model
# Load the HF_TOKEN from Hugging Face Secrets
hf_token = os.getenv("HF_TOKEN")

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Explicitly using float32 as requested
# Note: float32 works on both CPU and GPU
run_dtype = torch.float32

print(f"Using device: {device} with dtype: {run_dtype}")

# 2. Load the Pipeline
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=run_dtype,
        use_auth_token=hf_token
    )
    
    # --- ADDING DPM SOLVER MULTISTEP SCHEDULER ---
    # This allows for high-quality images in fewer steps (20-25)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Generation Logic
def generate_image(prompt, negative_prompt, steps, guidance_scale):
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a text prompt!")
    
    try:
        # Generate the image
        # With DPM Solver, 20-30 steps is usually the 'sweet spot'
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=guidance_scale
        ).images[0]
        return image
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# 4. Gradio UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Optimized Text-to-Image Generator")
    gr.Markdown("Generating images using Stable Diffusion v1.5 + DPM Solver Scheduler.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Image Prompt", 
                placeholder="Describe the image you want to generate...", 
                lines=3
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt (Optional)", 
                placeholder="What to exclude (e.g. blurry, low quality)", 
                lines=2
            )
            
            with gr.Row():
                # DPM Solver works so well that we can lower the default steps to 25
                steps = gr.Slider(minimum=10, maximum=50, value=25, step=1, label="Inference Steps")
                guidance = gr.Slider(minimum=1, maximum=20, value=7.5, step=0.5, label="Guidance Scale")
            
            with gr.Row():
                generate_btn = gr.Button("Generate Image", variant="primary")
                clear_btn = gr.Button("Clear")
        
        with gr.Column():
            output_img = gr.Image(label="Generated Result")

    # Examples for users
    gr.Examples(
        examples=[
            ["A majestic lion wearing a golden crown, cinematic lighting", "blurry, distorted", 25, 7.5],
            ["A cozy cabin in the woods during winter, oil painting style", "low resolution", 20, 8.0]
        ],
        inputs=[prompt, neg_prompt, steps, guidance]
    )

    # Event triggers
    generate_btn.click(
        fn=generate_image, 
        inputs=[prompt, neg_prompt, steps, guidance], 
        outputs=output_img
    )
    
    clear_btn.click(
        lambda: [None, None, 25, 7.5, None], 
        outputs=[prompt, neg_prompt, steps, guidance, output_img]
    )

# 5. Launch
if __name__ == "__main__":
    # For Hugging Face Spaces, share=True is not needed
    demo.launch()
