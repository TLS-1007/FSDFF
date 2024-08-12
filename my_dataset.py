import glob
import os

import hdf5storage as h5
import torch
import torch.utils.data as data
import numpy as np
from scipy.io import loadmat


class MYDataset(data.Dataset):
    def __init__(self, data_path):
        data_names = glob.glob(os.path.join(data_path, '*.mat'))
        self.keys = data_names

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        mat = h5.loadmat(self.keys[index])
        hyper = np.float32(np.array(mat['rad']))
        hyper = torch.Tensor(hyper)
        rgb = np.float32(np.array(mat['rgb']))
        rgb = torch.Tensor(rgb)
        return rgb, hyper


class HSI_MSI_Data1(data.Dataset):
    def __init__(self,path,R,training_size,stride,num):
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            # img1 = img1/img1.max()
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

         self.train_hrhs_all  = train_hrhs
         self.train_hrms_all  = train_hrms
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrms, train_hrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]

class HSI_MSI_Data2(data.Dataset):
    def __init__(self,path,R,training_size,stride,num):
         imglist=os.listdir(path)
         train_hrhs=[]
         train_hrms=[]
         for i in range(num):
            img=loadmat(path+imglist[i])
            img1=img["b"]
            # img1 = img1/img1.max()
            HRHSI=np.transpose(img1,(2,0,1))
            MSI=np.tensordot(R,  HRHSI, axes=([1], [0]))
            train_hrhs.append(HRHSI)
            train_hrms.append(MSI)
         train_hrhs=torch.Tensor(train_hrhs)
         train_hrms=torch.Tensor(train_hrms)

         self.train_hrhs_all  = train_hrhs
         self.train_hrms_all  = train_hrms
    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms= self.train_hrms_all[index, :, :, :]
        return train_hrms, train_hrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]