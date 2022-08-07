#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import *
from PIL import ImageOps

# In[66]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# In[77]:


class myTransforms:
    def __init__(self,scale):
        self.scale = scale
    # Defining __call__ method
    def __call__(self, pic): ###### Write sequeential transforms here.
        t_pic = self.downscale(pic)
        # t_pic = self.BicubicInterpolation(t_pic)
        return t_pic
    ### Tranform PIL Image to New PIL Image
    def downscale(self,img): #Here img is PIL IMAGE
        w,h= img.size
        h_new,w_new = h//self.scale,w//self.scale
        img_resize = img.resize((w_new,h_new))

        return img_resize #downscaled img
    def BicubicInterpolation(self,low_img): #Here low_img is PIL IMAGE
        w,h = low_img.size
        interpolated_img = low_img.resize((w*self.scale,h*self.scale),Image.BICUBIC)
#         plt.imshow(interpolated_img)
        return interpolated_img


# In[78]:

CROP_SIZE =32
class DataSetF(torch.utils.data.Dataset):
    def __init__(self,imgFolder,scale):
        super(DataSetF,self).__init__()
        self.scale=scale
        self.imageNames = [os.path.join(imgFolder,x) for x in os.listdir(imgFolder)]
        crop_size = CROP_SIZE - (CROP_SIZE % scale) # Valid crop size
        self.input_transform = transforms.Compose( [
                                                    transforms.CenterCrop(crop_size),
                                                    myTransforms(scale),
                                                    transforms.ToTensor(),
                                                    ])
        self.target_transform = transforms.Compose([
                                                    transforms.CenterCrop(crop_size),
                                                    transforms.ToTensor(),
                                                    ])
    def __getitem__(self,idx):
        # inp = Image.open(self.imageNames[idx]) ###33 channel Image
        inp = Image.open(self.imageNames[idx]).convert('YCbCr')
        inp,_,_ = inp.split()
        # inp = ImageOps.grayscale(inp)
     
        target = inp.copy()
        inp = self.input_transform(inp)
        target = self.target_transform(target)
        return inp,target
    def __len__(self):
        return len(self.imageNames)


# # ### Below is Testing for DataSetF Class

# # In[79]:


# mo = DataSetF('./Train',2)


# # In[80]:


# len(mo)


# # In[82]:


# def imgShow(tensor):
#     return transforms.ToPILImage()(tensor)


# # In[84]:


# plt.imshow(imgShow(mo[0][0]))


# # In[85]:


# plt.imshow(imgShow(mo[0][1]))


# # In[90]:


# mo[0][0].shape


# # In[92]:


# mo[0][1].shape


# # In[93]:


# io='./Train/t1.bmp'
# oo=3


# # In[96]:


# ten=transforms.ToTensor()(Image.open(io))
# plt.imshow(imgShow(ten))
# ten.shape


# # In[101]:


# plt.imshow(myTransforms(scale=10)(Image.open(io)))


# Downscale an image and then using bicubic for interpolation and applying CNN
