#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
possibly short term
- smooth prompt blending A -> B
- automatic prompt injection
- investigate better noise
- understand mem acid better
- smooth continuation mode
- objects floating around or being interactive

nice for cosyne
- physical objects

long term
- parallelization and stitching
"""



#%%`
import sys
sys.path.append('../')


from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
import time

from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers.utils import load_image
import random
import xformers
import triton
import lunar_tools as lt
from PIL import Image
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import random as rn
import numpy as np
import xformers
import triton
import cv2
import sys
from datasets import load_dataset
from prompt_engineer.PromptBlender import PromptBlender
from prompt_engineer.PromptManager import PromptManager
from tqdm import tqdm

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

shape_cam=(600,800) 
do_compile = False
use_community_prompts = True
use_modulated_unet = True
sz_renderwin = (512*2, 512*4)
resolution_factor = 5
base_w = 20
base_h = 15
# do_add_noise = True
negative_prompt = "blurry, bland, black and white, monochromatic"

gpu = "cuda"
device = torch.device("cuda")
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_device=gpu, torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to(gpu)
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device=gpu, torch_dtype=torch.float16, local_files_only=True)
pipe.vae = pipe.vae.cuda(gpu)
pipe.set_progress_bar_config(disable=True)

if do_compile:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
    pipe = compile(pipe, config)

blender = PromptBlender(pipe, 0) # cuda device index 
renderer = lt.Renderer(width=sz_renderwin[1], height=sz_renderwin[0])

noise_resolution_w = base_w*resolution_factor
noise_resolution_h = base_h*resolution_factor

cam_resolution_w = 1920
cam_resolution_h = 1080
speech_detector = lt.Speech2Text()

# noise
latents = blender.get_latents()
noise_img2img_orig = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda(gpu)

image_displacement_accumulated = 0

#%% LOOP

modulations = {}
if use_modulated_unet:
    def noise_mod_func(sample):
        noise =  torch.randn(sample.shape, device=sample.device, generator=torch.Generator(device=sample.device).manual_seed(1))
        return noise    
    
    modulations['noise_mod_func'] = noise_mod_func
    
prompt_decoder = 'fire'
prompt_embeds_decoder, negative_prompt_embeds_decoder, pooled_prompt_embeds_decoder, negative_pooled_prompt_embeds_decoder = blender.get_prompt_embeds(prompt_decoder, negative_prompt)
video_list = ['macumbas']
prompt_list = ['abstract purple blue and red shapes, skulls made of crystals, fungae sprouting from crevices, lizards, colorfoul alien flowers made of glass, 4K, unreal engine',
               'abstract purple blue and red shapes, fungae sprouting from crevices, 4K, unreal engine',
               'abstract purple blue and red light, fungae sprouting from crevices, 4K, unreal engine',
               'abstract purple blue and red light, modified racing cars, tunning, 4K, unreal engine',
               'macro photo of alien flowers made of glass, 4K, unreal engine',
               'high resolution portrait photo of minimalist designer masks, 4K',
               'macro photo of fungal growth patterns in high resolution, 4K'
               'macro photo of fungal growth patterns mixed with crystals and glass reflecting surfaces, 4K',
               'macro photo of fungal growth patterns mixed with crystals and glass reflecting surfaces, crystal skulls, 4K'
               'macro photo of racing car parts, purple blue and red neon lights, crystal skulls, 4K'
               ]
promptmanager = PromptManager(False, prompt_list)


for video in video_list:
    
    print(f"Processing Video: {video}")
    mr = lt.MovieReader(video+'.mp4')
    ms = lt.MovieSaver(video+'PROCESSED2.mp4', fps=mr.fps_movie)
    total_frames = mr.nmb_frames
    print(f"Number of frames: {total_frames}")
    prompt_change_interval = total_frames // len(promptmanager.prompts)
    
    nframe = np.asarray(Image.fromarray(mr.get_next_frame()).resize((cam_resolution_w,cam_resolution_h)))
    last_diffusion_image = np.uint8(nframe)
    last_cam_img_torch = None
    
    for frame_index in tqdm(range(total_frames)):
            
        noise_img2img_fresh = torch.randn((1,4,noise_resolution_h,noise_resolution_w)).half().cuda(gpu)
        noise_mixing = 0.1
        noise_img2img = blender.interpolate_spherical(noise_img2img_orig, noise_img2img_fresh, noise_mixing)
                
        if frame_index % prompt_change_interval == 0:
           # Change to the next prompt
           prompt = promptmanager.get_new_prompt()
           print(f"Current prompt: {prompt}")
           prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.get_prompt_embeds(prompt, negative_prompt)
                
        cam_img = np.asarray(Image.fromarray(mr.get_next_frame()).resize((cam_resolution_w,cam_resolution_h)))
        cam_img = cv2.resize(cam_img.astype(np.uint8), (cam_resolution_w, cam_resolution_h))
    
        strength = 0.5 
        num_inference_steps = 2
        
        cam_img_torch = torch.from_numpy(cam_img.copy()).to(latents.device).float()
        coef_noise =  0.05 
        t_rand = (torch.rand(cam_img_torch.shape, device=cam_img_torch.device)[:,:,0].unsqueeze(2) - 0.5) * coef_noise * 255 * 5
        cam_img_torch += t_rand
        cam_img_torch = torch.clamp(cam_img_torch, 0, 255)
        cam_img = cam_img_torch.cpu().numpy()
        modulations = None
        
        image = pipe(image=Image.fromarray(cam_img.astype(np.uint8)), 
                        latents=latents, num_inference_steps=num_inference_steps, strength=strength, 
                        guidance_scale=0.5, prompt_embeds=prompt_embeds, 
                        negative_prompt_embeds=negative_prompt_embeds, 
                        pooled_prompt_embeds=pooled_prompt_embeds, 
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, noise_img2img=noise_img2img, 
                        modulations=modulations).images[0]
            
        last_diffusion_image = np.array(image, dtype=np.float32)
        
        do_antishift = False
        if do_antishift:
            last_diffusion_image = np.roll(last_diffusion_image,-4,axis=0)
        
        # Render the image
        renderer.render(image)
        ms.write_frame(image)
        
    ms.finalize()