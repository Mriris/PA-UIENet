#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import pytorch_ssim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _Loss
from net.Ushape_Trans import *
# from dataset import prepare_data, Dataset
from net.utils import *
import cv2
import matplotlib.pyplot as plt
from utility import plots as plots, ptcolor as ptcolor, ptutils as ptutils, data as data
from loss.LAB import *
from loss.LCH import *
from torchvision.utils import save_image

# In[2]:


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# In[3]:


def split(img):
    output = []
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output


# In[4]:


dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda:0')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# torch.set_default_tensor_type(torch.FloatTensor)


# In[5]:


# Initialize generator 
generator = Generator().cuda()
generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth"))

# In[6]:


generator.eval()

# In[7]:


path = './test/input/'  # 要改
path_list = os.listdir(path)
path_list.sort(key=lambda x: int(x.split('.')[0]))
i = 1
for item in path_list:
    impath = path + item
    imgx = cv2.imread(path + item)
    imgx = cv2.resize(imgx, (256, 256))
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = np.array(imgx).astype(dtype)

    imgx = torch.from_numpy(imgx)
    imgx = imgx.permute(2, 0, 1).unsqueeze(0)
    imgx = imgx / 255.0
    # plt.imshow(imgx[0,:,:,:])
    # plt.show()
    imgx = Variable(imgx).cuda()
    # print(imgx.shape)
    output = generator(imgx)
    out = output[3].data
    # out=cv2.resize(output[3].data,(640,480))
    save_image(out, "./test/output/" + item, nrow=5, normalize=True)
    i = i + 1


# In[8]:


def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_mse(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse


# In[8]:


# In[9]:


path1 = './test/GT/'  # 要改
path2 = './test/output/'  # 要改
path_list = os.listdir(path1)
path_list.sort(key=lambda x: int(x.split('.')[0]))
PSNR = []

for item in path_list:
    impath1 = path1 + item
    impath2 = path2 + item
    imgx = cv2.imread(impath1)
    imgx = cv2.resize(imgx, (256, 256))
    imgy = cv2.imread(impath2)
    imgy = cv2.resize(imgy, (256, 256))
    # print(imgx.shape)
    psnr1 = compute_psnr(imgx[:, :, 0], imgy[:, :, 0])
    psnr2 = compute_psnr(imgx[:, :, 1], imgy[:, :, 1])
    psnr3 = compute_psnr(imgx[:, :, 2], imgy[:, :, 2])

    psnr = (psnr1 + psnr2 + psnr3) / 3.0

    PSNR.append(psnr)

# In[10]:


PSNR = np.array(PSNR)
print(PSNR.mean())
