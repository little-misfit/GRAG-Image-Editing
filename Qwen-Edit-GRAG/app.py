"""
Minimal Gradio wrapper for the given Qwen-Image-Edit inference script.

Features:
- Loads the model once and reuses it.
- Inputs: image, edit prompt, cond_b, cond_delta, optional model path.
- Matches your original settings (size 1024, steps=24, true_cfg_scale=4.0,
  fixed seed=42, and the same GRAG scale structure repeated 60 times).

Run:
  pip install gradio pillow torch
  # plus your project deps providing hacked_models/* and model weights
  python gradio_qwen_edit_minimal.py

Then open the local URL printed by Gradio.
"""

import os
from typing import Optional

import gradio as gr
import torch
from PIL import Image
from huggingface_hub import snapshot_download
import os
# --- your project imports (as in the original script) ---
from hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
from hacked_models.pipeline import QwenImageEditPipeline
from hacked_models.models import QwenImageTransformer2DModel
from hacked_models.utils import seed_everything

# -----------------------------
# Global state
# -----------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
_PIPELINE: Optional[QwenImageEditPipeline] = None
_LOADED_MODEL_PATH: Optional[str] = None


def _load_pipeline(model_path: str) -> QwenImageEditPipeline:
    """Load (or reuse) the pipeline for the given model_path."""
    global _PIPELINE, _LOADED_MODEL_PATH
    if _PIPELINE is not None and _LOADED_MODEL_PATH == model_path:
        return _PIPELINE

    # Set seed once (matches original)
    seed_everything(42)

    # Load components
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(model_path, "scheduler"), torch_dtype=_DTYPE
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        os.path.join(model_path, "transformer"), torch_dtype=_DTYPE
    )

    pipe = QwenImageEditPipeline.from_pretrained(
        model_path, torch_dtype=_DTYPE, scheduler=scheduler, transformer=transformer
    )

    pipe.set_progress_bar_config(disable=None)
    pipe.to(_DTYPE)
    pipe.to(_DEVICE)

    _PIPELINE = pipe
    _LOADED_MODEL_PATH = model_path
    return pipe


def _build_grag_scale(cond_b: float, cond_delta: float, repeats: int = 60):
    """Replicates your original GRAG schedule structure.

    Each element is: ((512, 1.0, 1.0), (4096, cond_b, cond_delta))
    """
    return [((512, 1.0, 1.0), (4096, cond_b, cond_delta))] * repeats


def predict(
    image: Image.Image,
    edit_prompt: str,
    cond_b: float,
    cond_delta: float,
    pipeline,
):
    if image is None or not edit_prompt:
        return None


    # Match original preprocessing
    input_image = image.convert("RGB").resize((1024, 1024))

    inputs = {
        "image": input_image,
        "prompt": edit_prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 24,
        "return_dict": False,
        "grag_scale": _build_grag_scale(cond_b, cond_delta, repeats=60),
    }

    with torch.inference_mode():
        image_batch, x0_images, saved_outputs = pipe(**inputs)

    # Return the first image (same as original save behavior)
    return image_batch[0]





model_dir = "Qwen-Image-Edit"
repo_id = "Qwen/Qwen-Image-Edit"

if not os.path.exists(model_dir) or not os.listdir(model_dir):
    snapshot_download(repo_id=repo_id, local_dir=model_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded to {model_dir}")
else:
    print(f"Model already exists at {model_dir}")



pipe = _load_pipeline(model_dir)


with gr.Blocks(title="Qwen Image Edit — Minimal GRAG Demo") as demo:
    gr.Markdown("# Qwen Image Edit — Minimal GRAG Demo\nUpload an image, enter your edit instruction, and set GRAG params.")


    with gr.Row():
        in_image = gr.Image(label="Input Image", type="pil")
        out_image = gr.Image(label="Edited Output", type="pil")

    edit_prompt = gr.Textbox(label="Edit Instruction", placeholder="e.g., Put a pair of black-framed glasses on him.")
    with gr.Row():
        cond_b = gr.Slider(label="cond_b", minimum=0.8, maximum=2.0, value=1.0, step=0.01)
        cond_delta = gr.Slider(label="cond_delta", minimum=0.8, maximum=2.0, value=1.0, step=0.01)

    run_btn = gr.Button("Run Edit")

    run_btn.click(
        fn=predict,
        inputs=[in_image, edit_prompt, cond_b, cond_delta, pipe],
        outputs=[out_image],
        api_name="run_edit",
    )

    gr.Markdown(
        """
**Notes**
- Uses fixed seed=42 and num_inference_steps=24 to match your script.
- Resizes the input to 1024×1024 before inference (as in your code).
- `grag_scale` is built as a list of length 60 with the same tuples.
- Automatically chooses CUDA if available; otherwise runs on CPU.
        """
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
