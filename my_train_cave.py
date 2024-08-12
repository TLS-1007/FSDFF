# -*- coding:utf-8 -*-
import argparse
import math
from math import pi

from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import torch_dct as DCT
import hdf5storage
import metrics
from my_dataset import *
from model import *
import time
import os

import pandas as pd
from mymodel import net


class change(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ycbcr_image):
        batch = ycbcr_image.shape[0]
        channel = ycbcr_image.shape[1]
        size = ycbcr_image.shape[2]
        stride = 2
        ycbcr_image = ycbcr_image.reshape(batch, channel, size // stride, stride, size // stride, stride).permute(0, 2,
                                                                                                                  4, 1,
                                                                                                                  3, 5)

        ycbcr_image = DCT.dct_2d(ycbcr_image, norm='None')
        ycbcr_image1 = ycbcr_image.reshape(batch, size // stride, size // stride, -1).permute(0, 3, 1, 2)  # out1

        return ycbcr_image1

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / (torch.abs(label+1e-6))
        mrae = torch.mean(error.view(-1))
        return mrae

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument("--dataset", type=str, default='cave', help='train dataset name')
# train/val/test 数据集路径
parser.add_argument("--train_data_path", type=str, default='D:\\code\\cave\\cave_train\\', help='train data path')
parser.add_argument("--val_data_path", type=str, default= 'D:\\code\\cave\\caveall\\', help='val data path')
#parser.add_argument("--test_data_path", type=str, default='/data2/pbm/ssr_code/NTIRE2020_clean_cut/test2', help='test data path')
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs")

parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--out_path", type=str, default='save_model', help='path log files')
parser.add_argument("--patch_size", type=int, default=64, help="patch size")
parser.add_argument("--stride", type=int, default=32, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


save_path = os.path.join(opt.out_path, opt.dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path1 = os.path.join(save_path, 'rmse')
if not os.path.exists(save_path1):
    os.makedirs(save_path1)

save_path2 = os.path.join(save_path, 'sam')
if not os.path.exists(save_path2):
    os.makedirs(save_path2)

save_path3 = os.path.join(save_path, 'loss')
if not os.path.exists(save_path3):
    os.makedirs(save_path3)

save_path4 = os.path.join(save_path, 'psnr')
if not os.path.exists(save_path4):
    os.makedirs(save_path4)



# 记录结果的的csv文件
df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'val_loss', 'val_rmse', 'val_psnr', 'val_sam'])  # 列名
df.to_csv(os.path.join(save_path, 'val_result_record.csv'), index=False)
df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'test_loss', 'test_rmse', 'test_psnr', 'test_sam'])  # 列名
df.to_csv(os.path.join(save_path, 'test_result_record.csv'), index=False)

R = create_F()



rmse_optimal = 10
sam_optimal = 10
val_loss_optimal = 1
psnr_optimal=38


decay_power = 1.5
num = 20 # 30
max_iteration=math.ceil(((512-opt.patch_size)//opt.stride+1)*((512-opt.patch_size)//opt.stride+1)*num/opt.batch_size)*opt.epoch

# maxiteration=math.ceil(((1040-training_size)//stride+1)*((1392-training_size)//stride+1)*num/BATCH_SIZE)*EPOCH
warm_iter = math.floor(max_iteration / 40)
print(max_iteration)


# 加载数据集
train_data = HSI_MSI_Data1(opt.train_data_path, R, 64, opt.stride, num=20)

train_loader = data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

val_data = HSI_MSI_Data2(opt.val_data_path, R, 64, opt.stride, num=12)
val_loader = data.DataLoader(dataset=val_data, batch_size=1, shuffle=False)




R_inv = np.linalg.pinv(R)

R_inv = torch.Tensor(R_inv)
R2 = torch.Tensor(R)
R2 = R2.cuda()
R=torch.Tensor(R).cuda()


cnn=net().cuda()
cg=change().cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iteration, eta_min=1e-6)

loss_func = nn.L1Loss(reduction='mean').cuda()
losssam = MyarcLoss().cuda()

# 模型参数初始化
for m in cnn.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)



step = 0
for epoch in range(opt.epoch):
    time1 = time.time()
    cnn.train()
    tbar = tqdm(train_loader, ncols=100)
    train_loss = AverageMeter()
    for epoch_step, (a2, a1) in enumerate(tbar):
        # 学习率更新设置
        lr = optimizer.param_groups[0]['lr']
        step = step + 1
        output = cnn(a2.cuda())
        loss1 = loss_func(output, a1.cuda())
        aaa1=cg(output)
        aaa2=cg(a1.cuda())
        loss2=loss_func(aaa1,aaa2)
        loss=loss1+0.1*loss2
        train_loss.update(loss1.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tbar.set_description('epoch:{}  lr:{}  loss:{}'.format(epoch + 1, lr, train_loss.avg))

    if epoch%100 == 0 or (epoch>=950 and epoch%10==0) :
        path = 'D:\\code\\cave\\caveall\\' #测试数据
        imglist = os.listdir(path)
        RMSE=[]
        PSNR = []
        test_path = 'D:\\code\\my\\test_result\\cave\\'  #测试结果保存
        for i in range(0, len(imglist)):
            cnn.eval()
            img = loadmat(path + imglist[i])
            img1 = img["b"]
            # img1=img1/img1.max()

            HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
            MSI = torch.tensordot(R.cpu(), HRHSI, dims=([1], [0]))
            MSI_1 = torch.unsqueeze(MSI, 0)

            with torch.no_grad():

                Fuse = reconstruction(cnn, R,  MSI_1.cuda(), 64, 31)

                Fuse = Fuse.cpu().detach().numpy()
            Fuse = np.squeeze(Fuse)
            Fuse = np.clip(Fuse, 0, 1)
            faker_hyper = np.transpose(Fuse, (1, 2, 0))
            print(faker_hyper.shape)
            a, b = metrics.rmse1(Fuse, HRHSI)
            RMSE.append(a)
            PSNR.append(b)
            test_data_path = os.path.join(test_path + imglist[i])
            hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
            hdf5storage.savemat(test_data_path, {'rea': img1}, format='7.3')
        print(np.mean(RMSE),np.mean(PSNR))

