import torch
import logging
logging.getLogger('tensorflow').disabled=True
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import pickle
from torchvision import transforms,datasets
from pytorch_wavelets import DWTForward, DWTInverse
import os
from roc_tpr import cal_roc_tpr
import torchvision
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser()
parser.add_argument('--seed','-seed',default=1,type=int)
parser.add_argument('--prep_size', default=896, type =int)
parser.add_argument('--patch_size', default=224, type = int)
parser.add_argument('--noise_level', default=0.1, type=float)
args = parser.parse_args()

device = 'cuda'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.eval()
model = model.to(device)
normalize = transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))

dino_transform = transforms.Compose([transforms.Resize((args.prep_size,args.prep_size),interpolation=Image.BICUBIC), transforms.ToTensor(),normalize])
#dino_transform = transforms.Compose([Rescale_dino(args.prep_size),transforms.ToTensor(), normalize])


dwt = DWTForward(J=2, wave='haar').to(device)
idwt = DWTInverse(wave='haar').to(device)


real_adm = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/adm/imagenet_ai_0508_adm/val/nature/", transform = dino_transform)
fake_adm = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/adm/imagenet_ai_0508_adm/val/ai/", transform = dino_transform)

real_big = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/biggan/imagenet_ai_0419_biggan/val/nature/", transform = dino_transform)
fake_big = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/biggan/imagenet_ai_0419_biggan/val/ai/", transform = dino_transform)

real_gli = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/glide/imagenet_glide/val/nature/", transform = dino_transform)
fake_gli = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/glide/imagenet_glide/val/ai/", transform = dino_transform)

real_mid = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/midjourney/imagenet_midjourney/val/nature/", transform = dino_transform)
fake_mid = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/midjourney/imagenet_midjourney/val/ai/", transform = dino_transform)

real_sd4 = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/nature/", transform = dino_transform)
fake_sd4 = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/ai/", transform = dino_transform)

real_sd5 = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val/nature/",transform=dino_transform)
fake_sd5 = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val/ai/",transform=dino_transform)

real_vqd = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/VQDM/imagenet_ai_0419_vqdm/val/nature/", transform = dino_transform)
fake_vqd = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/VQDM/imagenet_ai_0419_vqdm/val/ai/", transform = dino_transform)

real_wuk = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/wukong/wukong/val/nature/", transform = dino_transform)
fake_wuk = datasets.ImageFolder("/mnt/disks/sdb/dataset/genimage/wukong/wukong/val/ai/", transform = dino_transform)

bsize=1

real_adm_loader = DataLoader(real_adm,batch_size=bsize,shuffle=False)
fake_adm_loader = DataLoader(fake_adm,batch_size=bsize,shuffle=False)
real_big_loader = DataLoader(real_big,batch_size=bsize,shuffle=False)
fake_big_loader = DataLoader(fake_big,batch_size=bsize,shuffle=False)
real_gli_loader = DataLoader(real_gli,batch_size=bsize,shuffle=False)
fake_gli_loader = DataLoader(fake_gli,batch_size=bsize,shuffle=False)
real_mid_loader = DataLoader(real_mid,batch_size=bsize,shuffle=False)
fake_mid_loader = DataLoader(fake_mid,batch_size=bsize,shuffle=False)
real_sd4_loader = DataLoader(real_sd4,batch_size=bsize,shuffle=False)
fake_sd4_loader = DataLoader(fake_sd4,batch_size=bsize,shuffle=False)
real_sd5_loader = DataLoader(real_sd5,batch_size=bsize,shuffle=False)
fake_sd5_loader = DataLoader(fake_sd5,batch_size=bsize,shuffle=False)
real_vqd_loader = DataLoader(real_vqd,batch_size=bsize,shuffle=False)
fake_vqd_loader = DataLoader(fake_vqd,batch_size=bsize,shuffle=False)
real_wuk_loader = DataLoader(real_wuk,batch_size=bsize,shuffle=False)
fake_wuk_loader = DataLoader(fake_wuk,batch_size=bsize,shuffle=False)


def cal_metric(model, input_tensor):
    b,c,h,w = input_tensor.shape

    with torch.no_grad():
        #print(input_tensor.shape)
        zero_tensor = input_tensor.unfold(2,args.patch_size,args.patch_size).unfold(3,args.patch_size,args.patch_size)
        #print(zero_tensor.shape)
        input_tensor = zero_tensor.reshape([b,c,-1,args.patch_size,args.patch_size]).transpose(1,2)
        input_tensor = input_tensor.reshape([-1,c,args.patch_size,args.patch_size])



        yl, yh = dwt(input_tensor)  # yl: low-pass residual, yh: list of high-pass coefficients
        yl_zeros = torch.zeros_like(yl)
        pert_hf = idwt((yl_zeros, yh))
        perturbed_tensor = input_tensor - args.noise_level * pert_hf #/ torch.norm(pert_hf, dim=[2,3],keepdim=True)

        #perturbed_tensor = input_tensor + args.noise_level * torch.rand_like(input_tensor)
        outputs = model.forward_features(input_tensor, None)["x_norm_clstoken"]
        perturbed_outputs = model.forward_features(perturbed_tensor, None)["x_norm_clstoken"]

        similarity = torch.nn.functional.cosine_similarity(outputs, perturbed_outputs, dim=-1)

        similarity = similarity.unsqueeze(1).reshape([b,-1])



    similarity_min = torch.mean(similarity,1).view([b])
    index_min = torch.argmin(similarity,1).view(b)
    similarity_max = similarity.view([-1])
    index_max = torch.argmax(similarity,1).view(b)

    return similarity_min, similarity_max, index_min, index_max



def calnorm(loader):

    min_arr = []
    max_arr = []
    for batch_idx, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)

        with torch.no_grad():
            sub_min,sub_max,idx_min,idx_max = cal_metric(model, inputs)
            
        if batch_idx == 0:
            min_arr = sub_min.cpu().data.numpy()
            max_arr = sub_max.cpu().data.numpy()
            idx_min_arr = idx_min.cpu().data.numpy()
            idx_max_arr = idx_max.cpu().data.numpy()
            #norm_arr = sub.cpu().data.numpy()
        else:
            min_arr = np.concatenate((min_arr, sub_min.cpu().data.numpy()),0)
            max_arr = np.concatenate((max_arr, sub_max.cpu().data.numpy()),0)
            idx_min_arr = np.concatenate((idx_min_arr, idx_min.cpu().data.numpy()),0)
            idx_max_arr = np.concatenate((idx_max_arr, idx_max.cpu().data.numpy()),0)
            #norm_arr = np.concatenate((norm_arr,sub.cpu().data.numpy()),0)

    min_arr = np.reshape(min_arr, [-1])
    max_arr = np.reshape(max_arr, [-1])
    idx_min_arr = np.reshape(idx_min_arr, [-1])
    idx_max_arr = np.reshape(idx_max_arr, [-1])
    
    return np.array(min_arr), np.array(max_arr), np.array(idx_min_arr), np.array(idx_max_arr)
    #norm_arr = np.reshape(norm_arr,[-1])
    #return np.array(norm_arr)

print('adm')
radm0,radm1,radm2,radm3 = calnorm(real_adm_loader)
fadm0,fadm1,fadm2,fadm3 = calnorm(fake_adm_loader)
print('biggan')
rbig0,rbig1,rbig2,rbig3 = calnorm(real_big_loader)
fbig0,fbig1,fbig2,fbig3 = calnorm(fake_big_loader)
print('glide')
rgli0,rgli1,rgli2,rgli3 = calnorm(real_gli_loader)
fgli0,fgli1,fgli2,fgli3 = calnorm(fake_gli_loader)
print('midjourney')
rmid0,rmid1,rmid2,rmid3 = calnorm(real_mid_loader)
fmid0,fmid1,fmid2,fmid3 = calnorm(fake_mid_loader)
print('sd14')
rsd40,rsd41,rsd42,rsd43 = calnorm(real_sd4_loader)
fsd40,fsd41,fsd42,fsd43 = calnorm(fake_sd4_loader)
print('sd15')
rsd50,rsd51,rsd52,rsd53 = calnorm(real_sd5_loader)
fsd50,fsd51,fsd52,fsd53 = calnorm(fake_sd5_loader)
print('vqdm')
rvqd0,rvqd1,rvqd2,rvqd3 = calnorm(real_vqd_loader)
fvqd0,fvqd1,fvqd2,fvqd3 = calnorm(fake_vqd_loader)
print('wukong')
rwuk0,rwuk1,rwuk2,rwuk3 = calnorm(real_wuk_loader)
fwuk0,fwuk1,fwuk2,fwuk3 = calnorm(fake_wuk_loader)



folder_name = './paper_stats/genimage/wavelet_nonorm_l2/'+str(args.prep_size)+'/'+str(args.patch_size)+'/' + str(args.noise_level) +'/'
folder_min = folder_name+'min/'
folder_max = folder_name+'max/'


os.makedirs(folder_min,exist_ok=True)
np.savetxt(folder_min+'real_adm.txt',radm0)
np.savetxt(folder_min+"fake_adm.txt",fadm0)
np.savetxt(folder_min+'real_big.txt',rbig0)
np.savetxt(folder_min+"fake_big.txt",fbig0)
np.savetxt(folder_min+'real_gli.txt',rgli0)
np.savetxt(folder_min+"fake_gli.txt",fgli0)
np.savetxt(folder_min+'real_mid.txt',rmid0)
np.savetxt(folder_min+"fake_mid.txt",fmid0)
np.savetxt(folder_min+'real_sd4.txt',rsd40)
np.savetxt(folder_min+"fake_sd4.txt",fsd40)
np.savetxt(folder_min+'real_sd5.txt',rsd50)
np.savetxt(folder_min+"fake_sd5.txt",fsd50)
np.savetxt(folder_min+'real_vqd.txt',rvqd0)
np.savetxt(folder_min+"fake_vqd.txt",fvqd0)
np.savetxt(folder_min+'real_wuk.txt',rwuk0)
np.savetxt(folder_min+"fake_wuk.txt",fwuk0)


os.makedirs(folder_max,exist_ok=True)
np.savetxt(folder_max+'real_adm.txt',radm1)
np.savetxt(folder_max+"fake_adm.txt",fadm1)
np.savetxt(folder_max+'real_big.txt',rbig1)
np.savetxt(folder_max+"fake_big.txt",fbig1)
np.savetxt(folder_max+'real_gli.txt',rgli1)
np.savetxt(folder_max+"fake_gli.txt",fgli1)
np.savetxt(folder_max+'real_mid.txt',rmid1)
np.savetxt(folder_max+"fake_mid.txt",fmid1)
np.savetxt(folder_max+'real_sd4.txt',rsd41)
np.savetxt(folder_max+"fake_sd4.txt",fsd41)
np.savetxt(folder_max+'real_sd5.txt',rsd51)
np.savetxt(folder_max+"fake_sd5.txt",fsd51)
np.savetxt(folder_max+'real_vqd.txt',rvqd1)
np.savetxt(folder_max+"fake_vqd.txt",fvqd1)
np.savetxt(folder_max+'real_wuk.txt',rwuk1)
np.savetxt(folder_max+"fake_wuk.txt",fwuk1)

