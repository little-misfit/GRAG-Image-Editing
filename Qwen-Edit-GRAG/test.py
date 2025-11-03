import os
from PIL import Image
import torch
import sys
import json
import tqdm
from pathlib import Path
from termcolor import colored
from diffusers import QwenImageEditPipeline
from hacked_models.scheduler import FlowMatchEulerDiscreteScheduler
from hacked_models.pipeline import QwenImageEditPipeline
from hacked_models.models import QwenImageTransformer2DModel
import sys
import json
import tqdm
import argparse
from hacked_models.utils import *
seed_everything(42)

parser = argparse.ArgumentParser() 

parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit")
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--cond_b", type=float, required=True)
parser.add_argument("--cond_delta", type=float, required=True)



args = parser.parse_args()

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    os.path.join(args.model_path, "scheduler"),
    torch_dtype=torch.bfloat16,
)


transformer = QwenImageTransformer2DModel.from_pretrained(
    os.path.join(args.model_path, "transformer"),
    torch_dtype=torch.bfloat16,
)

pipeline = QwenImageEditPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                            scheduler = scheduler,
                                            transformer=transformer,
                                            )
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

DATA_ROOT = args.data_root
data_root = '/'.join(DATA_ROOT.split('/')[:-1])
out_path = os.path.join(args.out_path, DATA_ROOT.split('/')[-2],args.name)
os.makedirs(out_path,exist_ok=True)

with open (DATA_ROOT , 'r') as f:
    test_pairs = json.load(f)
print(colored(out_path,color = "green"))



for name , pair in tqdm.tqdm(list(test_pairs.items())):
    
    sample_name =  name
    img_folder, img_name = '/'.join(pair["image_path"].split('/')[:-1]) , pair["image_path"].split('/')[-1]
    # original_prompt = pair["original_prompt"]
    # editing_prompt = pair["editing_prompt"]
    editing_instruction = pair["editing_instruction"]

    input_image = Image.open(os.path.join(data_root,img_folder,img_name)).convert('RGB')
    os.makedirs(os.path.join(out_path,img_folder),exist_ok=True)
    os.makedirs(os.path.join(out_path,img_folder),exist_ok=True)
    if os.path.exists(os.path.join(out_path,img_folder,img_name)):
        print('Already generated.')
        continue
    grid = []
    prompt = editing_instruction
    inputs = {
        "image": input_image,
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 24,
        "return_dict": False,
        "grag_scale":[((512,1.0,1.0),(4096,args.cond_b,args.cond_delta))]*60,
    }

    with torch.inference_mode():
        output = pipeline(**inputs)
        image,x0_images,saved_outputs = output

    image[0].save(os.path.join(out_path,img_folder,img_name))
    