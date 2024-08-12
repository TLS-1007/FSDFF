import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch_dct as DCT
from thop import profile

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


class rechange(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, ycbcr_image1, ycbcr_image):
        batch = ycbcr_image1.shape[0]
        channel = ycbcr_image1.shape[1]

        size = ycbcr_image1.shape[2]
        stride = 2
        ycbcr_image2 = ycbcr_image.permute(0, 2, 3, 1).reshape(batch, size // stride, size // stride, channel, stride,
                                                               stride)
        ycbcr_image3 = DCT.idct_2d(ycbcr_image2, norm='None')
        ycbcr_image3 = ycbcr_image3.permute(0, 3, 1, 4, 2, 5)
        ycbcr_image4 = ycbcr_image3.reshape(batch, channel, size, size)

        return ycbcr_image4

class SKConv(nn.Module):
    def __init__(self, features):
        super(SKConv, self).__init__()

        self.features = features
        self.fc = nn.Linear(148, 148)
        self.fc1 = nn.Linear(148, 148)
        self.fc2 = nn.Linear(148, 124)
        self.fc3 = nn.Linear(148, 124)

        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, 148, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, 148, 3, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, x,x2):
        fea_U = x + x2
        fea_U = self.conv(fea_U)
        fea = x.unsqueeze_(dim=1)
        feas= x2.unsqueeze_(dim=1)
        feas = torch.cat([fea, feas], dim=1)


        fea1=self.conv3(fea_U).mean(-1).mean(-1)
        fea2 = self.conv4(fea_U).mean(-1).mean(-1)

        fea11=self.fc(fea1)
        fea11=self.fc2(fea11).unsqueeze_(dim=1)
        fea22 = self.fc1(fea2)
        fea22 = self.fc3(fea22).unsqueeze_(dim=1)
        attention_vectors = torch.cat([fea11,fea22], dim=1)




        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v






class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                    )


    def forward(self, x):
        out = self.layers(x)

        out = out + x
        return out


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # 创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h * w).transpose(1, 2), C.view(b, c, h * w)))
        E = torch.matmul(D.view(b, c, h * w), S.transpose(1, 2)).view(b, c, h, w)
        # gamma is a parameter which can be training and iter
        E = self.gamma * E + x

        return E


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h * w)).view(b, c, h, w)
        X = self.beta * X + x
        return X





class py(nn.Module):
    def __init__(self):
        super(py, self).__init__()
        self.fus=SKConv(124)
        self.cg=change()
        self.recg=rechange()
        self.channelatn=ChannelAttention()
        self.kjatn = PositionAttention(124)
        self.block1 = nn.Sequential(nn.Conv2d(124, 148, kernel_size=3, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    ResBlock(148),
                                    ChannelAttention(),
                                    ResBlock(148),
                                    ChannelAttention(),
                                    nn.Conv2d(148, 124, kernel_size=3, padding=1),
                                    ChannelAttention())
        self.block2 = nn.Sequential(nn.Conv2d(124, 148, kernel_size=3, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    ResBlock(148),
                                    PositionAttention(148),
                                    ResBlock(148),
                                    PositionAttention(148),
                                    nn.Conv2d(148, 124, kernel_size=3, padding=1),
                                    PositionAttention(124))


    def forward(self, x):
        x00=x
        x1=self.cg(x)
        x2=self.block1(x1)
        x3=self.block2(x1)
        #print(x3.shape)

        x4=self.fus(x2,x3)
        #print(x4.shape)

        x5=self.recg(x00,x4)

        return x5


class kj(nn.Module):
    def __init__(self):
        super(kj, self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(31, 62, kernel_size=3, padding=1),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  ResBlock(62),
                                  ChannelAttention(),
                                  ResBlock(62),
                                  ChannelAttention(),
                                  ResBlock(62),
                                  ChannelAttention(),
                                  nn.Conv2d(62, 31, kernel_size=3, padding=1),
                                  ChannelAttention())

    def forward(self, x):
        x1 = self.block1(x)
        return x1


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

# class SpaFre(nn.Module):
#     def __init__(self, channels=31):
#         super(SpaFre, self).__init__()
#
#         self.spa_att = nn.Sequential(nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=True),
#                                      nn.LeakyReLU(0.1),
#                                      nn.Conv2d(16, channels, kernel_size=3, padding=1, bias=True),
#                                      nn.Sigmoid())
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.contrast = stdv_channels
#         self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, 16, kernel_size=1, padding=0, bias=True),
#                                      nn.LeakyReLU(0.1),
#                                      nn.Conv2d(16, channels * 2, kernel_size=1, padding=0, bias=True),
#                                      nn.Sigmoid())
#         self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)
#
#     def forward(self, py, kj):  #, i
#         spafuse=py
#         frefuse=kj
#         spa_map = self.spa_att(spafuse-frefuse)
#         spa_res = frefuse*spa_map+spafuse
#         cat_f = torch.cat([spa_res,frefuse],1)
#         cha_res =  self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f))*cat_f)
#
#
#         return cha_res

class SpaFre(nn.Module):
    def __init__(self, channels=31):
        super(SpaFre, self).__init__()

        self.cg=change()
        self.recg=rechange()
        self.a1 = nn.Sequential(nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.a2 = nn.Sequential(nn.Conv2d(62, 31, kernel_size=1, padding=0),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True))


        self.pre_att = nn.Sequential(nn.Conv2d(124, 124, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                     ChannelAttention(),
                                     nn.Conv2d(124, 124, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                     ChannelAttention())
        self.spa_att = nn.Sequential(nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                     ChannelAttention(),
                                     nn.Conv2d(31, 31, kernel_size=3, padding=1),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True),

                                     ChannelAttention())

    def forward(self, py, kj):  #, i
        all1=kj+py
        all2=self.a1(all1)
        py1=self.cg(all2)
        py2=self.pre_att(py1)
        py3=self.recg(py,py2)*py
        all3=self.spa_att(all2)*kj
        fusion=torch.cat((py3,all3),dim=1)
        fusion=self.a2(fusion)









        return fusion



class allblock(nn.Module):
    def __init__(self):
        super(allblock, self).__init__()
        self.kjblock=kj()
        self.pyblock=py()
        self.fusion=SpaFre()
        self.bb=nn.Sequential(nn.Conv2d(31, 62, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                              nn.Conv2d(62, 31, kernel_size=3, padding=1, bias=True),
                              nn.LeakyReLU(0.1))

    def forward(self, x):
        x1 = self.kjblock(x)
        x2 = self.pyblock(x)
        out1=self.fusion(x2,x1)
        out2=self.bb(out1)
        out=out2+x


        return out


class net(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31):
        super(net, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.a1=allblock()
        self.a2 = allblock()
        self.a3 = allblock()
        self.a4 = allblock()
        self.a5 = allblock()
        self.a6 = allblock()
        self.a7 = allblock()






    def forward(self, x):
        x = self.conv_in(x)
        x1=self.a1(x)
        x2 = self.a2(x1)
        x3 = self.a3(x2)
        x4 = self.a4(x3)
        x5 = self.a5(x4)
        return x5

a=torch.rand(1,3,64,64)
cnn=net()
b=cnn(a)
print(b.shape)
flops, params = profile(cnn, inputs=(a, ))
print('flops:{}'.format(flops))
print('params:{}'.format(params))













