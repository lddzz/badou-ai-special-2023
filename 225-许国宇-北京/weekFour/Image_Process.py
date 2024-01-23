import math

import torch
import torch.nn as nn
import cv2
from torch.nn.functional import interpolate
from torch.nn import ConvTranspose2d
import numpy as np
from matplotlib import pyplot as plt
import random

from pca import PCA


class Canny:
    def __init__(self,src,kernel_size,sigma):
        '''

        :param src: source image
        :param kernel_size: Gaussian smooth kernel size
        :param sigma:Gaussian smoothing standard deviation
        '''
        imgpros=ImgageProcess()
        self.gray_iamge=imgpros.gray(src)
        self.smoothing_image=imgpros.gaussSmoothing(self.gray_iamge,sigma=sigma,kernel_size=kernel_size)
        self.gradient()
        self.nonMaximumSupp()
        self.doubleThresholdDetection()

    def gradient(self):
        sobel_kernel_horizontal=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_kernel_vertical=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        # stores the gradient image
        img_grad_x=np.zeros(self.gray_iamge.shape)
        img_grad_y=np.zeros([self.gray_iamge.shape[0],self.gray_iamge.shape[1]])
        img_grad=np.zeros(self.gray_iamge.shape)
        img_pad=np.pad(self.gray_iamge,((1,1),(1,1)),'constant')
        for r in range(self.gray_iamge.shape[1]):
            for c in range(self.gray_iamge.shape[0]):
                img_grad_x[r,c]=np.sum(img_pad[r:r+3,c:c+3]*sobel_kernel_horizontal)
                img_grad_y[r,c]=np.sum(img_pad[r:r+3,c:c+3]*sobel_kernel_vertical)
                img_grad[r,c]=np.sqrt(img_grad_x[r,c]**2+img_grad_y[r,c]**2)
        img_grad_x[img_grad_x==0]=0.00000001
        tan_angle=img_grad_y/img_grad_x
        self.img_grad=img_grad
        self.tan_angle=tan_angle
        plt.figure(2)
        plt.title('Gradient image')
        plt.imshow(self.img_grad.astype(np.uint8),cmap='gray')
        plt.axis('off')
        plt.show(block=True)

    def nonMaximumSupp(self):
        img_supp=np.zeros(self.img_grad.shape)
        for r in range(1,self.img_grad.shape[1]-1):
            for c in range(1,self.img_grad.shape[0]-1):
                # Marks whether the pixel is to be erased in the eight neighborhood
                flag=True
                # Gradient amplitude of the 8-neighborhood matrix
                eight_nearby_grad=self.img_grad[r-1:r+2,c-1:c+2]
                # The linear interpolation method is used to determine whether it is suppressed or not
                if self.tan_angle[r,c]<=-1:
                    num1=(eight_nearby_grad[0,1]-eight_nearby_grad[0,0])/self.tan_angle[r,c]+eight_nearby_grad[0,1]
                    num2=(eight_nearby_grad[2,1]-eight_nearby_grad[2,2])/self.tan_angle[r,c]+eight_nearby_grad[2,1]
                    if not(self.img_grad[r,c]>num1 and self.img_grad[r,c]>num2):
                        flag=False
                elif self.tan_angle[r,c]>=1:
                    num1 = (eight_nearby_grad[0, 2] - eight_nearby_grad[0, 1]) /self.tan_angle[r, c] + \
                           eight_nearby_grad[0, 1]
                    num2 = (eight_nearby_grad[2, 0] - eight_nearby_grad[2, 1]) / self.tan_angle[r, c] + \
                           eight_nearby_grad[2, 1]
                    if not (self.img_grad[r, c] > num1 and self.img_grad[r, c] > num2):
                        flag = False
                elif self.tan_angle[r,c]>0:
                    num1 = (eight_nearby_grad[0, 2] - eight_nearby_grad[1, 2]) *self.tan_angle[r, c] + \
                           eight_nearby_grad[1, 2]
                    num2 = (eight_nearby_grad[2, 0] - eight_nearby_grad[1, 0]) *self.tan_angle[r, c] + \
                           eight_nearby_grad[1, 0]
                    if not (self.img_grad[r, c] > num1 and self.img_grad[r, c] > num2):
                        flag = False
                elif self.tan_angle[r,c]<0:
                    num1 = (eight_nearby_grad[1, 0] - eight_nearby_grad[0, 0]) * self.tan_angle[r, c] + \
                           eight_nearby_grad[1, 0]
                    num2 = (eight_nearby_grad[1, 2] - eight_nearby_grad[2, 2]) * self.tan_angle[r, c] + \
                           eight_nearby_grad[1, 2]
                    if not (self.img_grad[r, c] > num1 and self.img_grad[r, c] > num2):
                        flag = False
                if flag:
                    img_supp[r,c]=self.img_grad[r,c]
        self.img_supp=img_supp
        plt.figure(3)
        plt.title('Non-Maximun Suppress pic')
        plt.imshow(self.img_supp.astype(np.uint8),cmap='gray')
        plt.axis('off')
        plt.show(block=True)
    def doubleThresholdDetection(self):
        lower_boundary=self.img_grad.mean()*0.5
        high_boundary=lower_boundary*3
        #weak edge storage
        weakEdge=[]
        # Regardless of boundary
        for r in range(1,self.img_supp.shape[1]-1):
            for c in range(1,self.img_supp.shape[0]-1):
                if self.img_supp[r,c]>=high_boundary:
                    # Definite edge
                    self.img_supp[r,c]=255
                elif self.img_supp[r,c]<=lower_boundary:
                    self.img_supp[r,c]=0
                else:
                    weakEdge.append([r,c])
        eight_near_index=[]
        for i in range(len(weakEdge)):
            #determine the current row and column
            r,c=weakEdge[i]
            #determine the index of the 8 neighborhood
            eight_near_index.append([r-1,c-1])
            eight_near_index.append([r-1,c])
            eight_near_index.append([r-1,c+1])
            eight_near_index.append([r,c-1])
            eight_near_index.append([r,c+1])
            eight_near_index.append([r+1,c-1])
            eight_near_index.append([r+1,c])
            eight_near_index.append([r+1,c+1])
            reserved_mark=False
            for index in range(8):
                if self.img_supp[eight_near_index[index][0],eight_near_index[index][1]]==255:
                    reserved_mark=True
            if reserved_mark==False:
                self.img_supp[r,c]=0
        plt.figure(4)
        plt.title('the result of doubleThresholdDetection pic')
        plt.imshow(self.img_supp.astype(np.uint8),cmap='gray')
        plt.axis('off')
        plt.show(block=True)



class ImgageProcess(nn.Module):
    def __init__(self):
        pass
    # edge detection
    def canny(self,src,kernel_size,sigma):
        Canny(src,kernel_size,sigma)
    #principal component analysis
    def pca(self,src,K):
        '''

        :param src: the input matrix or image
        :param K: the number of features retained by the first k order
        :return: A matrix of reduced dimensions
        '''
        img=cv2.imread(src)
        img=PCA(img,K)
        print('pca_src',img)
        return img
    #给图片增加椒盐噪声
    def pepperSaltNoise(self,src,percentage):
        img=self.gray(src)
        noiseNum=int(percentage*img.shape[0]*img.shape[1])
        noiseImg=img
        for index in range(noiseNum):
            randx=random.randint(1,img.shape[0]-1)
            randy=random.randint(1,img.shape[1]-1)
            if random.random()<=0.5:
                noiseImg[randx,randy]=0
            else:
                noiseImg[randx,randy]=255
        cv2.imshow('pepperSaltNoise_pic', img)
        cv2.waitKey(0)
        return noiseImg
    #给图片增加高斯噪声
    #一般情况下为给灰度图片增加噪声
    def gaussianNoise(self,src,means,sigma,percentage):
        img=self.gray(src)
        noiseNum=int(percentage*img.shape[0]*img.shape[1])
        noiseImg=img
        for index in range(noiseNum):
            #每次从图像中取一个随机像素点
            #randx表示选取像素的行
            #randy表示选取像素的列
            #不处理图像的边缘，随机数从1开始，最大范围-1
            randx=random.randint(1,img.shape[0]-1)
            randy=random.randint(1,img.shape[1]-1)
            noiseImg[randx,randy]=noiseImg[randx,randy]+random.gauss(means,sigma)
            #判断该点像素的灰度值情况，不合理的合理化
            if noiseImg[randx,randy]<0:
                noiseImg[randx,randy]=0
            elif noiseImg[randx,randy]>255:
                noiseImg[randx,randy]=255
        cv2.imshow('gaussianNoise_pic', img)
        cv2.waitKey(0)
        return noiseImg
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
    # gauss smooth function
    def gaussSmoothing(self,src,sigma,kernel_size):
        # determine if the kernel_size is odd
        if kernel_size%2==0:
            kernel_size+=1
        #stores Gaussian kernel elements
        gaussian_kernel_element=np.zeros([kernel_size,kernel_size])
        # generates a sequence number
        index=[i-kernel_size//1 for i in range(kernel_size)]
        coe1=1/(2*math.pi*sigma**2)
        coe2=-1/(2*sigma**2)
        # generatging Gaussian kernel
        for r in range(kernel_size):
            for c in range(kernel_size):
                gaussian_kernel_element[r,c]=coe1*math.exp(coe2*(index[r]**2+index[c]**2))
        gaussian_kernel_element=gaussian_kernel_element/gaussian_kernel_element.sum()
        #store the image after Gaussian smoothing
        w,h=src.shape
        img_new=np.zeros(src.shape)
        # margin padding
        padding_size=kernel_size//2
        img_padding=np.pad(src,((padding_size,padding_size),(padding_size,padding_size)),'constant')
        for r in range(h):
            for c in range(w):
                img_new[r,c]=np.sum(img_padding[r:r+kernel_size,c:c+kernel_size]*gaussian_kernel_element)

        plt.figure(1)
        plt.title('Gaussian smoothing pic')
        plt.imshow(img_new.astype(np.uint8),cmap='gray')
        plt.axis('off')
        plt.show(block=True)
        return img_new
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
    # imgpros.binaryzation(img,threshold=0.6)
    # imgpros.gaussianNoise(img,2,4,0.9)
    # imgpros.pepperSaltNoise(img,0.2)
    # imgpros.pca(img,2)
    imgpros.canny(img,5,0.5)
