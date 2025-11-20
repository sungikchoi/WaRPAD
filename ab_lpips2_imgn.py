import torch
import logging
logging.getLogger('tensorflow').disabled=True
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
import sys
import argparse
import numpy as np
from numba import jit
from PIL import Image
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pickle
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import lpips
from torchvision import transforms,datasets 
import os
matplotlib.use('agg')
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline,AutoPipelineForText2Image,VQDiffusionPipeline,StableDiffusionDiffEditPipeline
from  diffusers import AutoencoderKL,  DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler, DDIMScheduler, DDIMInverseScheduler
from roc_tpr import cal_roc_tpr
import torchvision
from utils import import_autoencoder, apply_filter, load_genimage
from torch.utils.data import DataLoader, Subset
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents

def check(vae, loader,filter_idx,vae_idx,kernel_size,sigma_space,sigma_color):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
    result = []
    for batch_idx, (inputs, targets) in enumerate(loader):
    
        inputs = inputs.to("cuda")
        batch_size = inputs.size(0)

        input_img = inputs * 2 -1
        if vae_idx == 0:
            latent = vae.encode(input_img).latent_dist.mode()
            image = vae.decode(latent)[0]
        else:
            latent = retrieve_latents(vae.encode(input_img))        
            image = vae.decode(latent,force_not_quantize=True)[0]
       
        output_img = image.to("cuda")
        
        for j_ in range(batch_size):
            sub,res = loss_fn_vgg(input_img[j_,:,:,:].unsqueeze(0),output_img[j_,:,:,:].unsqueeze(0),retPerLayer=True)
        

        result.append(res[1].cpu().data.numpy())
        
        inputs = inputs.detach()
        input_img = input_img.detach()
        latent = latent.detach()
        sub = sub.detach()
        output_img = output_img.detach()
        #res = res.detach()

    return np.reshape(np.array(result),[-1])


parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-seed', default=1557, type=int)
parser.add_argument('--ae', '-ae', choices = ['sd21', 'sd14', 'kadinsky','mini'], default='sd14', type = str)
parser.add_argument('--filter', '-filter', choices= ['bilateral_blur','blur_pool2d','box_blur','median_blur','gaussian_blur'], default = 'gaussian_blur', type=str)
parser.add_argument('--kernel_size', '-kernel_size', default = 3 ,type = int)
parser.add_argument('--sigma_space', '-sigma_space', default = 1.0, type = float)
parser.add_argument('--sigma_color', '-sigma_color', default = 0.1 ,type = float)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True


vae,vae_idx = import_autoencoder(args.ae)
real_adm,fake_adm,real_big,fake_big,real_gli,fake_gli,real_mid,fake_mid,real_sd4,fake_sd4,real_sd5,fake_sd5,real_vqd,fake_vqd,real_wuk,fake_wuk=load_genimage(args.ae)



real_adm_loader = torch.utils.data.DataLoader(real_adm, batch_size=1, shuffle=False)
fake_adm_loader = torch.utils.data.DataLoader(fake_adm, batch_size=1, shuffle=False)

real_big_loader = torch.utils.data.DataLoader(real_big, batch_size=1, shuffle=False)
fake_big_loader = torch.utils.data.DataLoader(fake_big, batch_size=1, shuffle=False)

real_gli_loader = torch.utils.data.DataLoader(real_gli, batch_size=1, shuffle=False)
fake_gli_loader = torch.utils.data.DataLoader(fake_gli, batch_size=1, shuffle=False)

real_mid_loader = torch.utils.data.DataLoader(real_mid, batch_size=1, shuffle=False)
fake_mid_loader = torch.utils.data.DataLoader(fake_mid, batch_size=1, shuffle=False)

real_sd4_loader = torch.utils.data.DataLoader(real_sd4, batch_size=1, shuffle=False)
fake_sd4_loader = torch.utils.data.DataLoader(fake_sd4, batch_size=1, shuffle=False)

real_sd5_loader = torch.utils.data.DataLoader(real_sd5, batch_size=1, shuffle=False)
fake_sd5_loader = torch.utils.data.DataLoader(fake_sd5, batch_size=1, shuffle=False)

real_vqd_loader = torch.utils.data.DataLoader(real_vqd, batch_size=1, shuffle=False)
fake_vqd_loader = torch.utils.data.DataLoader(fake_vqd, batch_size=1, shuffle=False)

real_wuk_loader = torch.utils.data.DataLoader(real_wuk, batch_size=1, shuffle=False)
fake_wuk_loader = torch.utils.data.DataLoader(fake_wuk, batch_size=1, shuffle=False)


real_adm_stats = check(vae, real_adm_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_adm_stats = check(vae, fake_adm_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('adm')
print(cal_roc_tpr(-real_adm_stats,-fake_adm_stats,0.95))

real_big_stats = check(vae, real_big_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_big_stats = check(vae, fake_big_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('big')
print(cal_roc_tpr(-real_big_stats,-fake_big_stats,0.95))

real_gli_stats = check(vae, real_gli_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_gli_stats = check(vae, fake_gli_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('gli')
print(cal_roc_tpr(-real_gli_stats,-fake_gli_stats,0.95))

real_mid_stats = check(vae, real_mid_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_mid_stats = check(vae, fake_mid_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('midjourney')
print(cal_roc_tpr(-real_mid_stats,-fake_mid_stats,0.95))

real_sd4_stats = check(vae, real_sd4_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_sd4_stats = check(vae, fake_sd4_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('sd 1.4')
print(cal_roc_tpr(-real_sd4_stats,-fake_sd4_stats,0.95))

real_sd5_stats = check(vae, real_sd5_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_sd5_stats = check(vae, fake_sd5_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('sd 1.5')
print(cal_roc_tpr(-real_sd5_stats,-fake_sd5_stats,0.95))

real_vqd_stats = check(vae, real_vqd_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_vqd_stats = check(vae, fake_vqd_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('vqdm')
print(cal_roc_tpr(-real_vqd_stats,-fake_vqd_stats,0.95))

real_wuk_stats = check(vae, real_wuk_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
fake_wuk_stats = check(vae, fake_wuk_loader, args.filter, vae_idx, args.kernel_size, args.sigma_space, args.sigma_color)
print('wukong')
print(cal_roc_tpr(-real_wuk_stats,-fake_wuk_stats,0.95))



save_dir = './paper_stats/genimage/aeroblade_lpips2/'
os.makedirs(save_dir,exist_ok=True)

np.savetxt(save_dir+str(args.ae)+'_'+'real_adm.txt',real_adm_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_adm.txt',fake_adm_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_big.txt',real_big_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_big.txt',fake_big_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_gli.txt',real_gli_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_gli.txt',fake_gli_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_mid.txt',real_mid_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_mid.txt',fake_mid_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_sd4.txt',real_sd4_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_sd4.txt',fake_sd4_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_sd5.txt',real_sd5_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_sd5.txt',fake_sd5_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_vqd.txt',real_vqd_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_vqd.txt',fake_vqd_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'real_wuk.txt',real_wuk_stats)
np.savetxt(save_dir+str(args.ae)+'_'+'fake_wuk.txt',fake_wuk_stats)



