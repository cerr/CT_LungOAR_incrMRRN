import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np



import torch.nn.functional as F
from torchvision import models


import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import torch as t

def normalize_data(data):
    data[data<24]=24
    data[data>1524]=1524
    
    data=data-24
    
    data=data*2./1500 - 1

    #data=(data+1)*1500./2
    return  data

def get_Incre_MRRN_Pytorch ():
    net=None
    
    net=Incre_MRRN_v2()
    #if use_gpu:
    net.cuda()
    

    return net

class Residual_Unit(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_c,inter_c,out_c):
        super(Residual_Unit, self).__init__()
        
        self.unit=CNN_block(in_c,inter_c)

    def forward(self, x):
        x_=self.unit(x)

        return x+x_

class CNN_block(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_c,inter_c):
        super(CNN_block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_c, inter_c, 3, 1, padding=1,bias=True)
        self.norm1=nn.BatchNorm2d(inter_c)
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        x2=self.norm1(x1)
        x3=self.activation(x2)

        return x3



class FRRU(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, in_c,inter_c,up_scale,adjust_channel,max_p_size):
        super(FRRU, self).__init__()
        
        self.maxp = nn.MaxPool2d(kernel_size=(max_p_size, max_p_size))
        self.drop=nn.Dropout2d(p=0.5)
        self.cnn_block=nn.Sequential(
            nn.Conv2d(in_c, inter_c, 3, 1, padding=1,bias=True),
            nn.BatchNorm2d(inter_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_c, inter_c, 3, 1,padding=1, bias=True),
            nn.BatchNorm2d(inter_c),
            nn.ReLU(inplace=True))
        self.channel_adjust=nn.Conv2d(inter_c, adjust_channel, 1, 1, padding=0,bias=True)
        self.upsample= nn.Upsample(scale_factor=up_scale,mode = "bilinear")
        


    def forward(self, p_s,r_s):
        r_s1=self.maxp(r_s)
		# Here can choose to not use
        #r_s1=self.drop(r_s1)

        #print ('r_s1 sizie ',r_s1.size())
        #print ('p_s sizie ',p_s.size())
        merged_=t.cat((r_s1,p_s),dim=1)
        pool_sm_out=self.cnn_block(merged_)
        adjust_out1=self.channel_adjust(pool_sm_out)
        adjust_out1_up_samp=self.upsample(adjust_out1)
        residual_sm_out=adjust_out1_up_samp+r_s

        return pool_sm_out,residual_sm_out


class Incre_MRRN(nn.Module):
    def __init__(self):
        super(Incre_MRRN, self).__init__()

        
        self.CNN_block1=CNN_block(3,32)
        self.RU1=Residual_Unit(32,32,32)
        self.RU2=Residual_Unit(32,32,32)
        self.RU3=Residual_Unit(32,32,32)

        self.RU11=Residual_Unit(32,32,32)
        self.RU22=Residual_Unit(32,32,32)
        self.RU33=Residual_Unit(32,32,32)


        self.Pool_stream1=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.5))

        self.Pool_stream2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.5))

        self.Pool_stream3=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.5))

        self.Pool_stream4=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(p=0.5))


        self.Residual_stream1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        self.FRRU1_1=FRRU(64,64,2,32,2)
        self.FRRU1_2=FRRU(96,64,2,32,2)
        self.FRRU1_3=FRRU(96,64,2,32,2)

        self.FRRU2_1=FRRU(96,128,4,32,4)
        self.FRRU2_2=FRRU(192,128,2,64,2)
        self.FRRU2_3=FRRU(160,128,4,32,4)

        self.FRRU3_1=FRRU(160,256,8,32,8)
        self.FRRU3_2=FRRU(384,256,2,128,2)
        self.FRRU3_3=FRRU(320,256,4,64,4)
        self.FRRU3_4=FRRU(288,256,8,32,8)

        self.FRRU4_1=FRRU(288,512,16,32,16)
        self.FRRU4_2=FRRU(768,512,2,256,2)
        self.FRRU4_3=FRRU(640,512,4,128,4)
        self.FRRU4_4=FRRU(576,512,8,64,8)
        self.FRRU4_5=FRRU(544,512,16,32,16)

        self.FRRU33_1=FRRU(288,256,8,32,8)
        self.FRRU33_2=FRRU(384,256,2,128,2)
        self.FRRU33_3=FRRU(320,256,4,64,4)
        self.FRRU33_4=FRRU(288,256,8,32,8)

        self.FRRU22_1=FRRU(160,128,4,32,4)
        self.FRRU22_2=FRRU(192,128,2,64,2)
        self.FRRU22_3=FRRU(160,128,4,32,4)

        self.FRRU11_1=FRRU(96,64,2,32,2)
        self.FRRU11_2=FRRU(96,64,2,32,2)
        self.FRRU11_3=FRRU(96,64,2,32,2)

        self.out_conv=nn.Conv2d(32, 2, 1, 1, bias=True)
        #self.out_act=nn.Sigmoid()
    def forward (self,x):
        x1= self.CNN_block1(x)   
        x2= self.RU1(x1)
        x3= self.RU2(x2)
        x4= self.RU3(x3)
        #print ('after 3RU ',x4.size())
        rs_2=self.Pool_stream1(x4)
        rs_1=self.Residual_stream1 (x4)

        rs_2,rs_1=self.FRRU1_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_3(rs_2,rs_1)
        #print ('after FRRU1_3 ',rs_2.size())
        #print ('after FRRU1_3 ',rs_1.size())
        rs_3= self.Pool_stream2(rs_2)

        rs_3,rs_1=self.FRRU2_1(rs_3,rs_1)
        rs_3,rs_2=self.FRRU2_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU2_3(rs_3,rs_1)

        rs_4= self.Pool_stream3(rs_3)
        rs_4,rs_1=self.FRRU3_1(rs_4,rs_1)
        rs_4,rs_3=self.FRRU3_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU3_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU3_4(rs_4,rs_1)

        rs_5= self.Pool_stream4(rs_4)

        rs_5,rs_1=self.FRRU4_1(rs_5,rs_1)
        rs_5,rs_4=self.FRRU4_2(rs_5,rs_4)
        rs_5,rs_3=self.FRRU4_3(rs_5,rs_3)
        rs_5,rs_2=self.FRRU4_4(rs_5,rs_2)
        rs_5,rs_1=self.FRRU4_5(rs_5,rs_1)

        rs_4,rs_1=self.FRRU33_1(rs_4,rs_1)
        rs_4,rs_3=self.FRRU33_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU33_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU33_4(rs_4,rs_1)

        rs_3,rs_1=self.FRRU22_1(rs_3,rs_1)
        rs_3,rs_2=self.FRRU22_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU22_3(rs_3,rs_1)

        rs_2,rs_1=self.FRRU11_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_3(rs_2,rs_1)

        rs_1= self.RU11(rs_1)
        rs_1= self.RU22(rs_1)
        rs_1= self.RU33(rs_1)
        #
        out=self.out_conv(rs_1)
        #_,tep,_,_=out.size()
        #if tep==1:
        #out=self.out_act(out)
        #print ('out size ',rs_1.size())
        return out
    
        

class Incre_MRRN_v2(nn.Module):
    def __init__(self):
        super(Incre_MRRN_v2, self).__init__()

        
        self.CNN_block1=CNN_block(1,32)
        self.CNN_block2=CNN_block(96,32)
        self.RU1=Residual_Unit(32,32,32)
        self.RU2=Residual_Unit(32,32,32)
        self.RU3=Residual_Unit(32,32,32)

        self.RU11=Residual_Unit(32,32,32)
        self.RU22=Residual_Unit(32,32,32)
        self.RU33=Residual_Unit(32,32,32)


        self.Pool_stream1=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream3=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream4=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream3=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream4=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))


        self.Residual_stream1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        self.FRRU1_1=FRRU(64,64,2,32,2)
        self.FRRU1_2=FRRU(96,64,2,32,2)
        self.FRRU1_3=FRRU(96,64,2,32,2)

        self.FRRU2_1=FRRU(96,128,4,32,4)
        self.FRRU2_2=FRRU(192,128,2,64,2)
        self.FRRU2_3=FRRU(160,128,4,32,4)

        self.FRRU3_1=FRRU(160,256,8,32,8)
        self.FRRU3_2=FRRU(384,256,2,128,2)
        self.FRRU3_3=FRRU(320,256,4,64,4)
        self.FRRU3_4=FRRU(288,256,8,32,8)

        self.FRRU4_1=FRRU(288,512,16,32,16)
        self.FRRU4_2=FRRU(768,512,2,256,2)
        self.FRRU4_3=FRRU(640,512,4,128,4)
        self.FRRU4_4=FRRU(576,512,8,64,8)
        self.FRRU4_5=FRRU(544,512,16,32,16)

        self.FRRU33_1=FRRU(544,256,8,32,8)
        self.FRRU33_2=FRRU(384,256,2,128,2)
        self.FRRU33_3=FRRU(320,256,4,64,4)
        self.FRRU33_4=FRRU(288,256,8,32,8)

        self.FRRU22_1=FRRU(288,128,4,32,4)
        self.FRRU22_2=FRRU(192,128,2,64,2)
        self.FRRU22_3=FRRU(160,128,4,32,4)

        self.FRRU11_1=FRRU(96,64,2,32,2)
        self.FRRU11_2=FRRU(96,64,2,32,2)
        self.FRRU11_3=FRRU(96,64,2,32,2)

        self.out_conv=nn.Conv2d(32, 7, 1, 1, bias=True)
        #self.out_act=nn.Sigmoid()
    def forward (self,x):
        x1= self.CNN_block1(x)   
        x2= self.RU1(x1)
        x3= self.RU2(x2)
        x4= self.RU3(x3)
        #print ('after 3RU ',x4.size())
        rs_2=self.Pool_stream1(x4)
        rs_1=self.Residual_stream1 (x4)

        rs_2,rs_1=self.FRRU1_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_3(rs_2,rs_1)
        #print ('after FRRU1_3 ',rs_2.size())
        #print ('after FRRU1_3 ',rs_1.size())
        rs_3= self.Pool_stream2(rs_2)

        rs_3,rs_1=self.FRRU2_1(rs_3,rs_1)
        rs_3,rs_2=self.FRRU2_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU2_3(rs_3,rs_1)

        rs_4= self.Pool_stream3(rs_3)
        rs_4,rs_1=self.FRRU3_1(rs_4,rs_1)
        rs_4,rs_3=self.FRRU3_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU3_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU3_4(rs_4,rs_1)

        rs_5= self.Pool_stream4(rs_4)

        rs_5,rs_1=self.FRRU4_1(rs_5,rs_1)
        rs_5,rs_4=self.FRRU4_2(rs_5,rs_4)
        rs_5,rs_3=self.FRRU4_3(rs_5,rs_3)
        rs_5,rs_2=self.FRRU4_4(rs_5,rs_2)
        rs_5,rs_1=self.FRRU4_5(rs_5,rs_1)

        #Start to do the up-sampling pool
        rs_5=self.Up_stream1(rs_5)
        rs_4,rs_1=self.FRRU33_1(rs_5,rs_1)
        rs_4,rs_3=self.FRRU33_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU33_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU33_4(rs_4,rs_1)

        rs_4=self.Up_stream2(rs_4)
        rs_3,rs_1=self.FRRU22_1(rs_4,rs_1)
        rs_3,rs_2=self.FRRU22_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU22_3(rs_3,rs_1)

        rs_3=self.Up_stream3(rs_3)
        rs_2,rs_1=self.FRRU11_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_3(rs_2,rs_1)

        rs_2=self.Up_stream1(rs_2)
        rs_1=torch.cat((rs_1,rs_2),dim=1)
        rs_1=self.CNN_block2(rs_1)

        rs_1= self.RU11(rs_1)
        rs_1= self.RU22(rs_1)
        rs_1= self.RU33(rs_1)
        #
        out=self.out_conv(rs_1)

        
        #out=F.sigmoid(out)
        
        return out,out,out
        

class Incre_MRRN_v2_SA_last(nn.Module):
    def __init__(self):
        super(Incre_MRRN_v2_SA_last, self).__init__()

        
        self.CNN_block1=CNN_block(1,32)
        self.CNN_block2=CNN_block(96,32)
        self.RU1=Residual_Unit(32,32,32)
        self.RU2=Residual_Unit(32,32,32)
        self.RU3=Residual_Unit(32,32,32)

        self.RU11=Residual_Unit(32,32,32)
        self.RU22=Residual_Unit(32,32,32)
        self.RU33=Residual_Unit(32,32,32)


        self.Pool_stream1=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream3=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream4=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream3=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream4=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))


        self.Residual_stream1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        self.FRRU1_1=FRRU(64,64,2,32,2)
        self.FRRU1_2=FRRU(96,64,2,32,2)
        self.FRRU1_3=FRRU(96,64,2,32,2)

        self.FRRU2_1=FRRU(96,128,4,32,4)
        self.FRRU2_2=FRRU(192,128,2,64,2)
        self.FRRU2_3=FRRU(160,128,4,32,4)

        self.FRRU3_1=FRRU(160,256,8,32,8)
        self.FRRU3_2=FRRU(384,256,2,128,2)
        self.FRRU3_3=FRRU(320,256,4,64,4)
        self.FRRU3_4=FRRU(288,256,8,32,8)

        self.FRRU4_1=FRRU(288,512,16,32,16)
        self.FRRU4_2=FRRU(768,512,2,256,2)
        self.FRRU4_3=FRRU(640,512,4,128,4)
        self.FRRU4_4=FRRU(576,512,8,64,8)
        self.FRRU4_5=FRRU(544,512,16,32,16)

        self.FRRU33_1=FRRU(544,256,8,32,8)
        self.FRRU33_2=FRRU(384,256,2,128,2)
        self.FRRU33_3=FRRU(320,256,4,64,4)
        self.FRRU33_4=FRRU(288,256,8,32,8)

        self.FRRU22_1=FRRU(288,128,4,32,4)
        self.FRRU22_2=FRRU(192,128,2,64,2)
        self.FRRU22_3=FRRU(160,128,4,32,4)

        self.FRRU11_1=FRRU(96,64,2,32,2)
        self.FRRU11_2=FRRU(96,64,2,32,2)
        self.FRRU11_3=FRRU(96,64,2,32,2)

        self.out_conv=nn.Conv2d(32, 12, 1, 1, bias=True)
        
        self.Block_SA1=Block_self_attention_inter_intra_change_last_layer(32,12,2,3)
        self.Block_SA2=Block_self_attention_inter_intra_change_last_layer(32,12,2,3)
        #self.out_act=nn.Sigmoid()
    def forward (self,x):
        x1= self.CNN_block1(x)   
        x2= self.RU1(x1)
        x3= self.RU2(x2)
        x4= self.RU3(x3)
        #print ('after 3RU ',x4.size())
        rs_2=self.Pool_stream1(x4)
        rs_1=self.Residual_stream1 (x4)

        rs_2,rs_1=self.FRRU1_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_3(rs_2,rs_1)
        #print ('after FRRU1_3 ',rs_2.size())
        #print ('after FRRU1_3 ',rs_1.size())
        rs_3= self.Pool_stream2(rs_2)

        rs_3,rs_1=self.FRRU2_1(rs_3,rs_1)
        rs_3,rs_2=self.FRRU2_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU2_3(rs_3,rs_1)

        rs_4= self.Pool_stream3(rs_3)
        rs_4,rs_1=self.FRRU3_1(rs_4,rs_1)
        rs_4,rs_3=self.FRRU3_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU3_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU3_4(rs_4,rs_1)

        rs_5= self.Pool_stream4(rs_4)

        rs_5,rs_1=self.FRRU4_1(rs_5,rs_1)
        rs_5,rs_4=self.FRRU4_2(rs_5,rs_4)
        rs_5,rs_3=self.FRRU4_3(rs_5,rs_3)
        rs_5,rs_2=self.FRRU4_4(rs_5,rs_2)
        rs_5,rs_1=self.FRRU4_5(rs_5,rs_1)

        #Start to do the up-sampling pool
        rs_5=self.Up_stream1(rs_5)
        rs_4,rs_1=self.FRRU33_1(rs_5,rs_1)
        rs_4,rs_3=self.FRRU33_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU33_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU33_4(rs_4,rs_1)

        rs_4=self.Up_stream2(rs_4)
        rs_3,rs_1=self.FRRU22_1(rs_4,rs_1)
        rs_3,rs_2=self.FRRU22_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU22_3(rs_3,rs_1)

        rs_3=self.Up_stream3(rs_3)
        rs_2,rs_1=self.FRRU11_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_3(rs_2,rs_1)

        rs_2=self.Up_stream1(rs_2)
        rs_1=torch.cat((rs_1,rs_2),dim=1)
        rs_1=self.CNN_block2(rs_1)
        
        rs_1=self.Block_SA1(rs_1)
        rs_1=self.Block_SA2(rs_1)
        rs_1= self.RU11(rs_1)
        rs_1= self.RU22(rs_1)
        rs_1= self.RU33(rs_1)
        #
        out=self.out_conv(rs_1)

        
        #out=F.sigmoid(out)
        
        return out,out,out
        

class Incre_MRRN_v2_SA_second(nn.Module):
    def __init__(self):
        super(Incre_MRRN_v2_SA_second, self).__init__()

        
        self.CNN_block1=CNN_block(1,32)
        self.CNN_block2=CNN_block(96,32)
        self.RU1=Residual_Unit(32,32,32)
        self.RU2=Residual_Unit(32,32,32)
        self.RU3=Residual_Unit(32,32,32)

        self.RU11=Residual_Unit(32,32,32)
        self.RU22=Residual_Unit(32,32,32)
        self.RU33=Residual_Unit(32,32,32)


        self.Pool_stream1=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream2=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream3=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Pool_stream4=nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream1=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream2=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream3=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))

        self.Up_stream4=nn.Sequential(
            nn.Upsample(scale_factor=2,mode = "bilinear"))#,
            #nn.Dropout2d(p=0.5))


        self.Residual_stream1 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        self.FRRU1_1=FRRU(64,64,2,32,2)
        self.FRRU1_2=FRRU(96,64,2,32,2)
        self.FRRU1_3=FRRU(96,64,2,32,2)

        self.FRRU2_1=FRRU(96,128,4,32,4)
        self.FRRU2_2=FRRU(192,128,2,64,2)
        self.FRRU2_3=FRRU(160,128,4,32,4)

        self.FRRU3_1=FRRU(160,256,8,32,8)
        self.FRRU3_2=FRRU(384,256,2,128,2)
        self.FRRU3_3=FRRU(320,256,4,64,4)
        self.FRRU3_4=FRRU(288,256,8,32,8)

        self.FRRU4_1=FRRU(288,512,16,32,16)
        self.FRRU4_2=FRRU(768,512,2,256,2)
        self.FRRU4_3=FRRU(640,512,4,128,4)
        self.FRRU4_4=FRRU(576,512,8,64,8)
        self.FRRU4_5=FRRU(544,512,16,32,16)

        self.FRRU33_1=FRRU(544,256,8,32,8)
        self.FRRU33_2=FRRU(384,256,2,128,2)
        self.FRRU33_3=FRRU(320,256,4,64,4)
        self.FRRU33_4=FRRU(288,256,8,32,8)

        self.FRRU22_1=FRRU(288,128,4,32,4)
        self.FRRU22_2=FRRU(192,128,2,64,2)
        self.FRRU22_3=FRRU(160,128,4,32,4)

        self.FRRU11_1=FRRU(96,64,2,32,2)
        self.FRRU11_2=FRRU(96,64,2,32,2)
        self.FRRU11_3=FRRU(96,64,2,32,2)

        self.out_conv=nn.Conv2d(32, 12, 1, 1, bias=True)
        
        self.Block_SA1=self.Block_SA2=Block_self_attention_inter_intra_change_second_layer(32,12,2,3)
        self.Block_SA2=self.Block_SA2=Block_self_attention_inter_intra_change_second_layer(32,12,2,3)
        #self.out_act=nn.Sigmoid()
    def forward (self,x):
        x1= self.CNN_block1(x)   
        x2= self.RU1(x1)
        x3= self.RU2(x2)
        x4= self.RU3(x3)
        #print ('after 3RU ',x4.size())
        rs_2=self.Pool_stream1(x4)
        rs_1=self.Residual_stream1 (x4)

        rs_2,rs_1=self.FRRU1_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU1_3(rs_2,rs_1)
        #print ('after FRRU1_3 ',rs_2.size())
        #print ('after FRRU1_3 ',rs_1.size())
        rs_3= self.Pool_stream2(rs_2)

        rs_3,rs_1=self.FRRU2_1(rs_3,rs_1)
        rs_3,rs_2=self.FRRU2_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU2_3(rs_3,rs_1)

        rs_4= self.Pool_stream3(rs_3)
        rs_4,rs_1=self.FRRU3_1(rs_4,rs_1)
        rs_4,rs_3=self.FRRU3_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU3_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU3_4(rs_4,rs_1)

        rs_5= self.Pool_stream4(rs_4)

        rs_5,rs_1=self.FRRU4_1(rs_5,rs_1)
        rs_5,rs_4=self.FRRU4_2(rs_5,rs_4)
        rs_5,rs_3=self.FRRU4_3(rs_5,rs_3)
        rs_5,rs_2=self.FRRU4_4(rs_5,rs_2)
        rs_5,rs_1=self.FRRU4_5(rs_5,rs_1)

        #Start to do the up-sampling pool
        rs_5=self.Up_stream1(rs_5)
        rs_4,rs_1=self.FRRU33_1(rs_5,rs_1)
        rs_4,rs_3=self.FRRU33_2(rs_4,rs_3)
        rs_4,rs_2=self.FRRU33_3(rs_4,rs_2)
        rs_4,rs_1=self.FRRU33_4(rs_4,rs_1)

        rs_4=self.Up_stream2(rs_4)
        rs_3,rs_1=self.FRRU22_1(rs_4,rs_1)
        rs_3,rs_2=self.FRRU22_2(rs_3,rs_2)
        rs_3,rs_1=self.FRRU22_3(rs_3,rs_1)

        rs_3=self.Up_stream3(rs_3)
        rs_2,rs_1=self.FRRU11_1(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_2(rs_2,rs_1)
        rs_2,rs_1=self.FRRU11_3(rs_2,rs_1)
        rs_2=self.Block_SA1(rs_2)
        rs_2=self.Block_SA2(rs_2)
		
        rs_2=self.Up_stream1(rs_2)
        rs_1=torch.cat((rs_1,rs_2),dim=1)
        rs_1=self.CNN_block2(rs_1)
        

        rs_1= self.RU11(rs_1)
        rs_1= self.RU22(rs_1)
        rs_1= self.RU33(rs_1)
        #
        out=self.out_conv(rs_1)

        
        #out=F.sigmoid(out)
        
        return out,out,out
        
class Position_AM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(Position_AM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        #out = self.gamma*out + x
        out = out + x
        return out
        
class Block_self_attention_inter_intra_change_last_layer(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=32,block_width=16,stride=2,kernel=3):
        super(Block_self_attention_inter_intra_change_last_layer, self).__init__()
        self.chanel_in = in_dim

        self.block_width=block_width
        self.inter_block_SA=Position_AM_Module(32)
        self.softmax = nn.Softmax(dim=-1)
        
        self.block_num=256/self.block_width
        self.stride=stride
        self.kernel=kernel
        self.split_size_H=[]
        self.split_size_W=[]
        for k in range (int(self.block_num)):
            self.split_size_H.append(self.block_width)
            self.split_size_W.append(self.block_width)
        self.scane_x_max_num=256/(self.block_width*self.stride)
        self.scane_y_max_num=self.scane_x_max_num#256/(self.block_width*self.stride)

    def forward(self, x):

        #print (x.size())
        #m_batchsize, C, height, width = x.size()

        #splited_chunk_H=torch.split(x,self.split_size_H,2)
        #splited_chunk=[]
        #for splited_chunk_H_tp in splited_chunk_H:
        #    splited_chunk.append(torch.split(splited_chunk_H_tp,self.split_size_W,3))
        
        x_clone=x.clone()   
        #print ('x block number is ', self.scane_x_max_num)
        #print ('x block number is ', self.scane_y_max_num)
        for i in range(int(self.scane_x_max_num)+1):
            for j in range (int(self.scane_y_max_num)+1):
                #print ('i is ',i)
                #print ('j is ',j)
                start_x=i*self.block_width*self.stride
                
                end_x=i*self.block_width*self.stride+self.block_width*self.kernel
                start_y=j*self.block_width*self.stride
                end_y=j*self.block_width*self.stride+self.block_width*self.kernel


                #assert (start_x<256)
                #assert (start_y<256)
                #end_y=torch.min(end_y,256)
                #end_x=torch.min(end_x,256)
                if end_y>256:
                    end_y=256
                if end_x>256:
                    end_x=256

                #print ('start_x: ',start_x)
                #print ('end_x: ',end_x)
                #print ('start_y: ',start_y)
                #print ('end_y: ',end_y)

                if start_x<256 and start_y<256:
                    x_clone[:,:,start_x:end_x,start_y:end_y]=self.inter_block_SA(x[:,:,start_x:end_x,start_y:end_y])

        return x_clone      
		
class Block_self_attention_inter_intra_change_second_layer(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=64,block_width=16,stride=2,kernel=3):
        super(Block_self_attention_inter_intra_change_second_layer, self).__init__()
        self.chanel_in = in_dim

        self.block_width=block_width
        self.inter_block_SA=Position_AM_Module(64)
        self.softmax = nn.Softmax(dim=-1)
        
        self.block_num=128/self.block_width
        self.stride=stride
        self.kernel=kernel
        self.split_size_H=[]
        self.split_size_W=[]
        for k in range (int(self.block_num)):
            self.split_size_H.append(self.block_width)
            self.split_size_W.append(self.block_width)
        self.scane_x_max_num=128/(self.block_width*self.stride)
        self.scane_y_max_num=self.scane_x_max_num#256/(self.block_width*self.stride)

    def forward(self, x):

        #print (x.size())
        #m_batchsize, C, height, width = x.size()

        #splited_chunk_H=torch.split(x,self.split_size_H,2)
        #splited_chunk=[]
        #for splited_chunk_H_tp in splited_chunk_H:
        #    splited_chunk.append(torch.split(splited_chunk_H_tp,self.split_size_W,3))
        
        x_clone=x.clone()   
        #print ('x block number is ', self.scane_x_max_num)
        #print ('x block number is ', self.scane_y_max_num)
        for i in range(int(self.scane_x_max_num)+1):
            for j in range (int(self.scane_y_max_num)+1):
                #print ('i is ',i)
                #print ('j is ',j)
                start_x=i*self.block_width*self.stride
                
                end_x=i*self.block_width*self.stride+self.block_width*self.kernel
                start_y=j*self.block_width*self.stride
                end_y=j*self.block_width*self.stride+self.block_width*self.kernel


                #assert (start_x<256)
                #assert (start_y<256)
                #end_y=torch.min(end_y,256)
                #end_x=torch.min(end_x,256)
                if end_y>128:
                    end_y=128
                if end_x>128:
                    end_x=128

                #print ('start_x: ',start_x)
                #print ('end_x: ',end_x)
                #print ('start_y: ',start_y)
                #print ('end_y: ',end_y)

                if start_x<128 and start_y<128:
                    x_clone[:,:,start_x:end_x,start_y:end_y]=self.inter_block_SA(x[:,:,start_x:end_x,start_y:end_y])

        return x_clone      		
