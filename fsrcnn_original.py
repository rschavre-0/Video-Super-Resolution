#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import *
from datafs import *
import math


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as DL
import torch.nn.functional as F
import torch.nn.init as init
torch.cuda.empty_cache()


# In[3]:


from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import copy

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# In[4]:


# data_f = './'
# train_path = data_f + 'DIV2K/DIV2K_valid_HR/'
# # train_path = data_f + 'Images/'
# test__path = data_f + 'Test/Set5/'


# In[5]:


scale =  3


# In[6]:


# train_set = DataSetF(train_path,scale) ## input Image folder and scale
# test_set =  DataSetF(test__path,scale)


# # In[7]:


# train_loader = DL(train_set) ## input Image folder and scale
# test_loader =  DL(test_set)


# In[8]:


## UTILITY FUNCTIONS
def Test_with_psnr(model):
    avg_psnr = 0
    for batch in test_loader:
        img,label = batch[0].to(device),batch[1].to(device)
        out = model(img)
        psnr = PSNR(out,label)
        avg_psnr += psnr
    print("Avg. PSNR :",{avg_psnr / len(test_loader)}, "dB.")
    return avg_psnr / len(test_loader)
    
def PSNR(imageTensor,labelTensor):
    criterion = nn.MSELoss()
    loss = criterion(imageTensor,labelTensor)
    psnr = 10*math.log10(1/loss.item())
    return psnr
def PSNR_img(image,label):
    assert image.size==label.size
    tensor = transforms.ToTensor()
    imageTensor = tensor(image)
    labelTensor = tensor(label)
    return PSNR(imageTensor,labelTensor)


# In FSRCNN, 5 main steps as in the figure with more convolutions are involved:
# 
#     1.Feature Extraction: Bicubic interpolation in previous SRCNN is replaced by 5×5 conv.
#     2.Shrinking: 1×1 conv is done to reduce the number of feature maps from d to s where s<<d.
#     3.Non-Linear Mapping: Multiple 3×3 layers are to replace a single wide one
#     4.Expanding: 1×1 conv is done to increase the number of feature maps from s to d.
#     5.Deconvolution: 9×9 filters are used to reconstruct the HR image.

# In[9]:


class FSRCNN(nn.Module):
    def __init__(self,upscale_factor):
        super(FSRCNN, self).__init__()
        # Feature extraction
        self.feature_extraction = nn.Sequential(
                      nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
                      nn.PReLU(56)
                      )
        # Shrinking
        self.shrink = nn.Sequential(
                      nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
                      nn.PReLU(12)
                      )
        
        # Mapping layer.
        self.map = nn.Sequential(
                      nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
                      nn.PReLU(12),
                      nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
                      nn.PReLU(12),
                      nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
                      nn.PReLU(12),
                      nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
                      nn.PReLU(12)
                      )

        # Expanding layer.
        self.expand = nn.Sequential(
                      nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
                      nn.PReLU(56)
                      )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(56, 1, (9, 9), (upscale_factor, upscale_factor), (4, 4), (upscale_factor - 1, upscale_factor - 1))

        # Initialize model weights.
        self._initialize_weights()

        
#     def forward_feed(self,x):
    def __call__(self,x):
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


# In[10]:


def Enhance_Image_FSRCNN(model,image,multichannel = False):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    converter = torch.load(model).to(device)
    img = Image.open(image) ###low Res Image not interpolated
    img = img.convert('YCbCr')
    # img = myTransforms(scale).BicubicInterpolation(img)  #We don't need to use bicubic for FSRCNN
    Y,Cb,Cr = img.split()
    tensor_img = transforms.ToTensor()
    
    channel1 = tensor_img(Y).view(1,-1,Y.size[1],Y.size[0]).to(device)
    out_channel1 = converter(channel1)
    out_channel1 = out_channel1.cpu()
    Y_enhanced =  np.array(out_channel1[0].detach())*255.
    Y_enhanced = Y_enhanced.clip(0,255)
    Y_enhanced = Image.fromarray(np.uint8(Y_enhanced[0]))
    
    channel2 = tensor_img(Cb).view(1,-1,Cb.size[1],Cb.size[0]).to(device)
    out_channel2 = converter(channel2)
    out_channel2 = out_channel2.cpu()
    Cb_enhanced =  np.array(out_channel2[0].detach())*255.
    Cb_enhanced = Cb_enhanced.clip(0,255)
    Cb_enhanced = Image.fromarray(np.uint8(Cb_enhanced[0]))
    
    channel3 = tensor_img(Cr).view(1,-1,Cr.size[1],Cr.size[0]).to(device)
    out_channel3 = converter(channel3)
    out_channel3 = out_channel3.cpu()
    Cr_enhanced =  np.array(out_channel3[0].detach())*255.
    Cr_enhanced = Cr_enhanced.clip(0,255)
    Cr_enhanced = Image.fromarray(np.uint8(Cr_enhanced[0]))
    
    res = Image.merge('YCbCr',[Y_enhanced,Cb_enhanced,Cr_enhanced]).convert('RGB')
    return res


# In[11]:


def TrainFSR(model,number_of_epoch):
    lsv=[]
    psnrval=[]
    
    model = model.to(device)
    criterion = nn.MSELoss()
#     criterion = MS_SSIM_L1_LOSS()
    optimizer = optim.Adam(  # we use Adam instead of SGD like in the paper, because it's faster
    [
        {"params": model.feature_extraction.parameters()},
        {"params": model.shrink.parameters()},
        {"params": model.map.parameters()},
        {"params": model.expand.parameters()},
        {"params": model.deconv.parameters(), "lr": 1e-4},
    ], lr=1e-4,
    )
    for epoch in range(number_of_epoch):
        loss_in_epoch = 0
        for iteration,batch in enumerate(train_loader):
            img,label = batch[0].to(device),batch[1].to(device)
            
            optimizer.zero_grad()
            
            out_img = model(img)
#             print(label.shape,out_img.shape)
            loss = criterion(out_img,label)
            loss.backward()
            optimizer.step()
            loss_in_epoch += loss.item()
            
        print(f"Epoch {epoch}. Training loss: {loss_in_epoch/ len(train_loader)}")
        lsv += [loss_in_epoch/ len(train_loader)]
        psnrval += [Test_with_psnr(model)]
    torch.save(model,f"fsrcnn_model_scale3.pth")
    return lsv,psnrval


# In[12]:


# len(train_set)


# In[13]:


# model = FSRCNN(scale)
# lsv,psnrval = TrainFSR(model,500)


# In[14]:


# plt.scatter(range(len(lsv)),lsv)
# plt.xlabel("Number of epochs")
# plt.ylabel("Loss Value ")
# plt.title("MSE variation using Adam Optimiser")
# plt.savefig("Variation of MSE")


# In[15]:


# plt.style.use('seaborn')
# plt.plot(psnrval)
# plt.xlabel("Number of epochs")
# plt.ylabel("Avg. PSNR Value(in dB)")
# plt.savefig("Variation of PSNR")


# In[16]:


# model = 'fsrcnn_model_scale3.pth'
# image = 'bird_GT.bmp'

# low_res = myTransforms(scale).downscale(Image.open(image))
# low_res.save('Low_Res_FSRCNN.bmp')
# interpolated_img = myTransforms(scale).BicubicInterpolation(low_res)
# interpolated_img.save("Interpolated_FSRCNN.bmp")
# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# In[17]:


# img = Enhance_Image_FSRCNN(model,'Low_Res_FSRCNN.bmp',2)
# img.save('Final_image_FSRCNN.bmp')
# plt.imshow(img)


# In[18]:


# low_res.size
#
#
# # In[19]:
#
#
# interpolated_img.size
#
#
# # In[20]:
#
#
# img.size


# In[21]:


# Image.open(image).size


# In[22]:


# PSNR_img(interpolated_img,Image.open(image))


# In[23]:


# PSNR_img(Image.open(image),img)


# In[24]:


# Image.open(image).size,img.size


# In[ ]:





# In[ ]:




