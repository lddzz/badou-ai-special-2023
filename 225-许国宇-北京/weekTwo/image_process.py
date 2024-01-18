import torch
import torch.nn as nn
import cv2
from torch.nn.functional import interpolate
from torch.nn import ConvTranspose2d
import numpy as np
from matplotlib import pyplot as plt

class ImgageProcess(nn.Module):
    def __init__(self):
        pass
    def gray(self,src):
        img=cv2.imread(src)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_pic',img)
        cv2.waitKey(0)
        return img
    def binaryzation(self,src,threshold=0.5):
        img=self.gray(src)
        rows,cols=img.shape
        for r in range(rows):
            for c in range(cols):
                if(img[r,c]<=threshold*256):
                    img[r,c]=0
                else:
                    img[r,c]=255
        cv2.imshow('binaryzation',img)
        cv2.waitKey(0)
        return img
    def handle_gray(self):
        pass
    def sample(self,src,mode,dst_size=None,align_corners=None):
        img=cv2.imread(src)
        img=torch.Tensor(img)
        img=torch.permute(img,(2,0,1))
        if mode=='linear':
            dst_size=dst_size[0]
            img = interpolate(img, size=dst_size, mode=mode, align_corners=align_corners)
        else:
            img=torch.unsqueeze(img,0)
            img=interpolate(img,size=dst_size,mode=mode,align_corners=align_corners)
            img=torch.squeeze(img)
        img=torch.permute(img,(1,2,0))
        img=img.numpy()
        img=img.astype(np.uint8)
        cv2.imshow(mode,img)
        cv2.waitKey(0)
        return img
    def conv2dTranspose(self,src):
        # src=cv2.imread(src)
        # img=torch.tensor(src)
        # img=torch.permute(img,(2,0,1))
        # conv2d_T=ConvTranspose2d(3,3,5,2,padding=0,output_padding=0)
        # img=conv2d_T(img)
        # img=torch.permute(img,(1,2,0))
        # img=img.numpy()
        # img=img.astype(np.uint8)
        # cv2.imshow('Conv2dTranspose',img)
        # cv2.waitKey(0)
        # return img
        pass
    def histc(self,src):
        img=self.gray(src)
        plt.figure()
        plt.hist(img.ravel(),256)
        plt.show()
    def equalHistc(self,src):
        img=cv2.imread(src)
        if len(img.shape)==3:
            (b,g,r)=cv2.split(img)
            bhistc=cv2.equalizeHist(b)
            ghistc=cv2.equalizeHist(g)
            rhistc=cv2.equalizeHist(r)
            result=cv2.merge([bhistc,ghistc,rhistc])
            cv2.imshow("equalhist",result)
            cv2.waitKey(0)
            return result
        elif len(img.shape)==2:
            dst=cv2.equalizeHist(img)
            cv2.imshow('equalhistc',dst)
            cv2.waitKey(0)
            return dst
    def cropping(self):
        pass
    def load(self):
        pass
    def learning(self):
        pass




if __name__=='__main__':
    imgpros=ImgageProcess()
    src='./data/test_pic/self.jpg'
    img='./data/test_pic/lenna.png'
    # imgpros.gray(src)
    # imgpros.sample(src,'nearest',(580,580),None)
    # imgpros.sample(src,'bilinear',(600,600),None)
    # imgpros.sample(src,'area',(1000,1000),None)
    # imgpros.sample(src,'linear',(700,700),None)
    # imgpros.sample(src,'nearest',(700,500),None)
    # imgpros.conv2dTranspose(img)
    # imgpros.histc(src)
    # imgpros.equalHistc(img)
    imgpros.binaryzation(img,threshold=0.6)