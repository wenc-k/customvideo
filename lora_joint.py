"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

import argparse
from typing import Literal
import os, sys
from safetensors import safe_open
from safetensors.torch import save_file

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
    CogVideoXPipeline_origin,
)

from diffusers.utils import export_to_video, load_image, load_video
from contextlib import contextmanager
import json

# 用于log保存
@contextmanager
def tee_stdout(filename):
    original_stdout = sys.stdout
    with open(filename, 'a') as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)

    def flush(self):
        for f in self.files:
            f.flush()

def merge_lora_weights(lora_path1, lora_path2, output_path, scale1=1.0, scale2=1.0):
    merged_weights = {}
    
    # 加载第一个LoRA
    with safe_open(lora_path1, framework="pt") as f:
        for key in f.keys():
            merged_weights[key] = f.get_tensor(key) * scale1
    
    # 加载第二个LoRA并叠加
    with safe_open(lora_path2, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key) * scale2
            if key in merged_weights:
                merged_weights[key] += tensor
            else:
                merged_weights[key] = tensor
    
    # 保存合并后的权重
    save_file(merged_weights, output_path)

def load_prompts(file_path: str) -> list[str]:
    """读取存储 prompts 的 txt 文件，返回字符串列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 逐行读取并去除换行符，过滤空行
        return [line.strip() for line in f if line.strip()]

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    use_dynamic_cfg: bool = True,
    lora_path_motion: str = None,
):
    os.makedirs(output_path, exist_ok=True)

    pipe = CogVideoXPipeline_origin.from_pretrained(model_path, torch_dtype=dtype) # base model

    # If you're using with lora, add this code
    if lora_path and lora_path_motion:
        print("\033[1;31m use joint lora inference ...... \033[0m")
        print("\033[1;31m id_lora_path: \033[0m", lora_path)
        print("\033[1;31m motion_lora_path:  \033[0m", lora_path_motion)
        merge_lora_path = os.path.join(output_path, "merged_lora.safetensors")
        # 合并权重（需调整scale系数）
        merge_lora_weights(
            lora_path1=os.path.join(lora_path, "pytorch_lora_weights.safetensors"),
            lora_path2=os.path.join(lora_path_motion, "pytorch_lora_weights.safetensors"),
            output_path=merge_lora_path,
            scale1=0.5,
            scale2=0.5
        )
        
        
        # 加载合并后的LoRA
        pipe.load_lora_weights(merge_lora_path, adapter_name="merged")
        pipe.set_adapters(["merged"])

    #pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.to("cuda")
    #pipe.enable_sequential_cpu_offload()
    #pipe.vae.enable_slicing()
    #pipe.vae.enable_tiling()
    prompt_list = load_prompts(prompt)

    json_datas_list = []
    video_dir_path = os.path.join(output_path, "videos")
    os.makedirs(video_dir_path, exist_ok=True)
    for i, prompt in enumerate(prompt_list):
        print(f"\033[1;31m Generating {i+1}/{len(prompt_list)}: {prompt} \033[0m")
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=False,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]
        current_file_path = os.path.join(video_dir_path, f"video_case_{i+1}.mp4")
        export_to_video(video_generate, current_file_path, fps=8)
        print("\033[1;31m saved video to: \033[0m", current_file_path)
        json_data = {
            "file_path": os.path.abspath(current_file_path),
            "prompt": prompt,
        }
        json_datas_list.append(json_data)

    # save json
    flie_prompt_json = os.path.join(output_path, "prompt.json")
    with open(flie_prompt_json, "w") as f:
        json.dump(json_datas_list, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="../CogVideoX-5b", help="The path of the pre-trained model to be used"    # change if use 5b-i2v
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')" # change if use 5b-i2v
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"   # change if use 5b-i2v
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument(
        "--use_dynamic_cfg",
        type=bool,
        default=True, 
    )
    parser.add_argument("--inf_lora_json_path", type=str, required=True, help="The path of the LoRA weights to be used")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16


    inf_lora_json_path = args.inf_lora_json_path
    with open(inf_lora_json_path, 'r') as f:
        inf_lora_json = json.load(f)

    for inf_case in inf_lora_json:
        args.output_path = inf_case['output_path']
        args.prompt = inf_case['prompt_path']
        args.lora_path = inf_case["lora_id_path"]
        args.lora_path_motion = inf_case["lora_motion_path"]

        generate_video(
            prompt=args.prompt,
            model_path=args.model_path,
            lora_path=args.lora_path,
            lora_rank=args.lora_rank,
            output_path=args.output_path,
            image_or_video_path=args.image_or_video_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos_per_prompt,
            dtype=dtype,
            generate_type=args.generate_type,
            seed=args.seed,
            use_dynamic_cfg=args.use_dynamic_cfg,
            lora_path_motion=args.lora_path_motion
        )
