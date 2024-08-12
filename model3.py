# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:32:24 2020

@author: Dian
"""
import math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import misc
import cv2
from scipy.io import loadmat
import sys
from matplotlib.pyplot import *
from numpy import *
from torch.nn import functional as F
from torch import nn
import torch
import os

import torch.utils.data as data
import metrics
import hdf5storage
import h5py



class HSI_MSI_Data3(data.Dataset):
    def __init__(self,path,R,training_size,stride,num,data_name='oldcomputer'):
        if data_name=='oldcomputer':
            save_data(path=path,R=R,training_size=training_size,stride=stride,num=num,data_name=data_name)
            data_path = 'E:\super-resolution\spectraldata\cave_patchdata\\'
        elif data_name == 'Harved':
            save_data(path=path,R=R,training_size=training_size,stride=stride,num=num,data_name=data_name)
            data_path = 'E:\super-resolution\spectraldata\Harved_patchdata\\'
        else:
            raise Exception("Invalid mode!", data_name)
        imglist=os.listdir(data_path)
        self.keys = imglist
        self.keys.sort()
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5py.File(self.keys[index], 'r')
        hyper = np.float32(np.array(mat['rad']))
#        hyper = np.transpose(hyper, [2, 1, 0])
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
#        rgb = np.transpose(rgb, [2, 1, 0])
        rgb = torch.Tensor(rgb)
        mat.close()
        return rgb, hyper

    
            

def reconstruction(net2,R,R_inv,MSI,training_size,stride):
        index_matrix=torch.zeros((R.shape[1],MSI.shape[2],MSI.shape[3])).cuda()
        abundance_t=torch.zeros((R.shape[1],MSI.shape[2],MSI.shape[3])).cuda()
        a=[]
        for j in range(0, MSI.shape[2]-training_size+1, stride):
            a.append(j)
        a.append(MSI.shape[2]-training_size)
        b=[]
        for j in range(0, MSI.shape[3]-training_size+1, stride):
            b.append(j)
        b.append(MSI.shape[3]-training_size)
        for j in a:
            for k in b:
                temp_hrms = MSI[:,:,j:j+training_size, k:k+training_size]

#                temp_hrms=torch.unsqueeze(temp_hrms, 0)
#                 print(temp_hrms.shape)
                with torch.no_grad():
                    # print(temp_hrms.shape)
#                    HSI = net2(R,R_inv,temp_hrms)
                    HSI = net2(temp_hrms)
                    HSI=HSI.squeeze()
#                   print(HSI.shape)
                    HSI=torch.clamp(HSI,0,1)
                    abundance_t[:,j:j+training_size, k:k+training_size]= abundance_t[:,j:j+training_size, k:k+training_size]+ HSI
                    index_matrix[:,j:j+training_size, k:k+training_size]= 1+index_matrix[:,j:j+training_size, k:k+training_size]
                
        HSI_recon=abundance_t/index_matrix
        return HSI_recon     
def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F
class HSI_MSI_Data(data.Dataset):
    def __init__(self,train_hrhs_all,train_hrms_all):
        self.train_hrhs_all  = train_hrhs_all
        self.train_hrms_all  = train_hrms_all
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms
    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data1(data.Dataset):
    def __init__(self,path,R,training_size,stride,num):
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            HRHSI=np.transpose(img1,(2,0,1))
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = train_hrhs
         self.train_hrms_all  = train_hrms
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]
class HSI_MSI_Data2(data.Dataset): 
    def __init__(self,path,R,training_size,stride,num):      
         imglist=os.listdir(path)
         train_hrhs=[]
         # train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=[]
         # train_hrms=torch.Tensor(train_hrms)
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["ref"]
            img1=img1/img1.max()
#            HRHSI=np.transpose(img1,(2,0,1))
#            MSI=np.tensordot(R, HRHSI, axes=([1], [0]))
            HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
            MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
            HRHSI = HRHSI.numpy()
            MSI = MSI.numpy()
            for j in range(0, HRHSI.shape[1]-training_size+1, stride):
                for k in range(0, HRHSI.shape[2]-training_size+1, stride):
                    temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
                    temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)
#         print(train_hrhs.shape, train_hrms.shape)
         self.train_hrhs_all  = torch.Tensor(train_hrhs)
         self.train_hrms_all  = torch.Tensor(train_hrms)
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrhs, train_hrms

    def __len__(self):
        return self.train_hrhs_all.shape[0]



def dataacquire(path,R,training_size,stride,num):      
  imglist=os.listdir(path)
  train_hrhs=[]
  train_hrms=[]
  for i in range(num):
    img=loadmat(path+imglist[i])
    img1=img["ref"]
    img1=img1/img1.max()
    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
    HRHSI = HRHSI.numpy()
    MSI = MSI.numpy()
    for j in range(0, HRHSI.shape[1]-training_size+1, stride):
        for k in range(0, HRHSI.shape[2]-training_size+1, stride):
            temp_hrhs = HRHSI[:,j:j+training_size, k:k+training_size]
            temp_hrms = MSI[:,j:j+training_size, k:k+training_size]
            train_hrhs.append(temp_hrhs)
            train_hrms.append(temp_hrms)
  return train_hrhs,train_hrms
     






def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warm_lr_scheduler(optimizer, init_lr1,init_lr2, warm_iter,iteraion, lr_decay_iter, max_iter, power):
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer
    if iteraion < warm_iter:
        lr=init_lr1+iteraion/warm_iter*(init_lr2-init_lr1)
    else:
      lr = init_lr2*(1 - (iteraion-warm_iter)/(max_iter-warm_iter))**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out



def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F

class LossTrainCSS(nn.Module):
    def __init__(self):
        super(LossTrainCSS, self).__init__()
       

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / (label+1e-10)
        mrae = torch.mean(error)
        return mrae
#
#    def mrae_loss(self, outputs, label):
#        error = torch.abs(outputs - label) / label
#        mrae = torch.mean(error)
#        return mrae

    def rgb_mrae_loss(self, outputs, label):
        error = torch.abs(outputs - label)
        mrae = torch.mean(error.view(-1))
        return mrae
 
    
class MyarcLoss(torch.nn.Module):
    def __init__(self):
        super(MyarcLoss, self).__init__()

    def forward(self, output, target):
        sum1=output*target
        sum2=torch.sum(sum1,dim=0)+1e-10
        norm_abs1=torch.sqrt(torch.sum(output*output,dim=0))+1e-10
        norm_abs2=torch.sqrt(torch.sum(target*target,dim=0))+1e-10
        aa=sum2/norm_abs1/norm_abs2
        aa[aa<-1]=-1
        aa[aa>1]=1
        spectralmap=torch.acos(aa)
        return torch.mean(spectralmap)
     

        
        