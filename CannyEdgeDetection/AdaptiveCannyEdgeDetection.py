#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:34:10 2022

@author: yingchihlin
"""

import argparse
import cv2
import math
import numpy as np
import time
from scipy.signal import convolve2d as conv2
from matplotlib import pyplot as plt
from ProcFunc import *


class CannyEdgeDetection:
    def __init__(self, img, smooth_method = 'Bilateral', 
                 threshold_mode = 'auto', l_threshold_ratio = 0.05, u_threshold_ratio = 0.15):
        self.img = img
        #Obtain the time
        self.stime = time.time()
        self.angles = [0,45,90,135]
        self.nang = len(self.angles)
        self.smooth_method = smooth_method
        self.threshold_mode = threshold_mode
        #self.l_threshold = l_threshold
        #self.u_threshold = u_threshold
        self.highThreshold = u_threshold_ratio
        self.lowThreshold = l_threshold_ratio
        self.result = {}
        
    def deroGauss(self, w=5, s=1, angle=0):
        wlim = (w-1)/2
        y,x = np.meshgrid(np.arange(-wlim,wlim+1),np.arange(-wlim,wlim+1))
        G = np.exp(-np.sum((np.square(x),np.square(y)),axis=0)/(2*np.float64(s)**2))
        G = G/np.sum(G)
        dGdx = -np.multiply(x,G)/np.float64(s)**2
        dGdy = -np.multiply(y,G)/np.float64(s)**2
        angle = angle*math.pi/180
        dog = math.cos(angle)*dGdx + math.sin(angle)*dGdy
        return dog
    
    def bilatfilt(self, I, w, sd, sr):
        dim = I.shape
        Iout= np.zeros(dim)
        wlim = (w-1)//2
        y,x = np.meshgrid(np.arange(-wlim,wlim+1),np.arange(-wlim,wlim+1))
        g = np.exp(-np.sum((np.square(x),np.square(y)),axis=0)/(2*(np.float64(sd)**2)))
        Ipad = np.pad(I,(wlim,),'edge')
        for r in range(wlim,dim[0]+wlim):
            for c in range(wlim,dim[1]+wlim):
                Ix = Ipad[r-wlim:r+wlim+1,c-wlim:c+wlim+1]
                s = np.exp(-np.square(Ix-Ipad[r,c])/(2*(np.float64(sr)**2)))
                k = np.multiply(g,s)
                Iout[r-wlim,c-wlim] = np.sum(np.multiply(k,Ix))/np.sum(k)
        return Iout

    # define function
    # Use the graussian filter to denoise
    def get_edges(self, I, sd):
        dim = I.shape
        Idog2d = np.zeros((self.nang,dim[0],dim[1]))
        for i in range(self.nang):
            dog2d = self.deroGauss(5,sd, self.angles[i])
            Idog2dtemp = abs(conv2(I,dog2d,mode='same',boundary='fill'))
            Idog2dtemp[Idog2dtemp<0]=0
            Idog2d[i,:,:] = Idog2dtemp
        return Idog2d

    # compute the gradient
    def calc_sigt(self, I,threshval):
        M,N = I.shape
        ulim = np.uint8(np.max(I))	
        N1 = np.count_nonzero(I>threshval)
        N2 = np.count_nonzero(I<=threshval)
        w1 = np.float64(N1)/(M*N)
        w2 = np.float64(N2)/(M*N)
        try:
            u1 = sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N1 for i in range(threshval+1,ulim))
            u2 = sum(i*np.count_nonzero(np.multiply(I>i-0.5,I<=i+0.5))/N2 for i in range(threshval+1))
            
            uT = u1*w1+u2*w2
            sigt = w1*w2*(u1-u2)**2
        except:
            return 0
        return sigt

    # NMS
    def nonmaxsup(self, I,gradang):
        dim = I.shape
        Inms = np.zeros(dim)
        xshift = int(np.round(math.cos(gradang*np.pi/180)))
        yshift = int(np.round(math.sin(gradang*np.pi/180)))
        Ipad = np.pad(I,(1,),'constant',constant_values = (0,0))
        for r in range(1,dim[0]+1):
            for c in range(1,dim[1]+1):
                maggrad = [Ipad[r-xshift,c-yshift],Ipad[r,c],Ipad[r+xshift,c+yshift]]
                if Ipad[r,c] == np.max(maggrad):
                    Inms[r-1,c-1] = Ipad[r,c]
        return Inms
    
    # Obtain the best threshold
    def get_threshold(self, I):
        max_sigt = 0
        opt_t = 0
        ulim = np.uint8(np.max(I))
        print(ulim,)
        for t in range(ulim+1):
            sigt = self.calc_sigt(I,t)
            if sigt > max_sigt:
                max_sigt = sigt
                opt_t = t
        print ('optimal high threshold: ',opt_t,)
        return opt_t
    
    # Threshold
    def threshold(self, I, uth):
        if self.threshold_mode == 'auto':
            lth = uth/2.5
            Ith = np.zeros(I.shape)
            Ith[I>=uth] = 255
            Ith[I<lth] = 0
            Ith[np.multiply(I>=lth, I<uth)] = 100
        else:
            highThreshold = I.max() * self.highThreshold
            lowThreshold = highThreshold * self.lowThreshold
            
            Ith = np.zeros(I.shape)
            Ith[I >= highThreshold] = 255
            Ith[I < lowThreshold] = 0
            Ith[np.multiply(I>=lowThreshold, I<highThreshold)] = 100
            
        return Ith
            
    # hysteresis
    def hysteresis(self, I):
        r,c = I.shape
        Ipad = np.pad(I,(1,),'edge')
        c255 = np.count_nonzero(Ipad==255)
        imgchange = True
        for i in range(1,r+1):
            for j in range(1,c+1):
                if Ipad[i,j] == 100:
                    if np.count_nonzero(Ipad[r-1:r+1,c-1:c+1]==255)>0:
                        Ipad[i,j] = 255
                    else:
                        Ipad[i,j] = 0
        Ih = Ipad[1:r+1,1:c+1]
        return Ih

    def detector(self):
        self.result['Original'] = self.img
        # Resize the image
        while self.img.shape[0] > 1100 or self.img.shape[1] > 1100:
            self.img = cv2.resize(self.img,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
        # translate into gray scale
        gimg = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        #Obtain the size of image
        dim = self.img.shape

        ## Start canny
        #Smoothing
        if self.smooth_method == 'Bilateral':
            #Bilateral filtering
            print ('Bilateral filtering...')
            gimg = self.bilatfilt(gimg, 5, 3, 10)
            print ('after bilat: ',np.max(gimg))
        elif self.smooth_method == 'Gaussain':
            print ('Gaussain filtering...')
            gimg = gs_filter(gimg)
            print ('after bilat: ',np.max(gimg))

        self.result['Smoothed'] = gimg

        #Gradient of Image
        print ('Calculating Gradient...')
        img_edges = self.get_edges(gimg,2)
        print ('after gradient: ',np.max(img_edges))

        #Non-max suppression：（NMS）
        print ('Suppressing Non-maximas...')
        for n in range(self.nang):
            img_edges[n,:,:] = self.nonmaxsup(img_edges[n,:,:], self.angles[n])

        print ('after nms: ', np.max(img_edges),)

        img_edge = np.max(img_edges,axis=0)
        lim = np.uint8(np.max(img_edge))
        
        self.result['NMS'] = img_edge

        # Compute the threshold
        print ('Calculating Threshold...')
        th = self.get_threshold(gimg)
        the = self.get_threshold(img_edge)

        # Obtain the best threshold
        print ('Thresholding...')
        img_edge = self.threshold(img_edge, the*0.25)

        print ('Applying Hysteresis...')
        img_edge = self.nonmaxsup(self.hysteresis(img_edge),90)

        print( 'Time taken :: ', str(time.time()-self.stime)+' seconds...')

        self.result['Result'] = img_edge
        # cv2.imwrite(img_name+'3'+'.jpg', img_edge)
        # cv2.imwrite(img_name+'4'+'.jpg', img_canny)

        return self.result

    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-image1" , type = str , default='Face.jpg')
    parser.add_argument("-image2" , type = str , default='rose.tif')
    parser.add_argument("-image3" , type = str , default='Koa2.jpg')
    parser.add_argument("-image4" , type = str , default='car2.jpg')
    parser.add_argument("-image5" , type = str , default='jet.png')
    args = parser.parse_args()

    image = cv2.imread(args.image5)


    result = {}
    # #My implementation
    detector = CannyEdgeDetection(image, smooth_method = 'Gaussain', 
                 threshold_mode = 'auto', l_threshold_ratio = 0.15, u_threshold_ratio = 0.10)
    result_auto = detector.detector()
    result['Adaptation Method'] = result_auto['Result']

    #Manual
    detector = CannyEdgeDetection(image, smooth_method = 'Gaussain', 
                 threshold_mode = 'Manual', l_threshold_ratio = 0.15, u_threshold_ratio = 0.10)
    result_m = detector.detector()
    result['Manual'] = result_m['Result']

    # #OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.medianBlur(gray, 11)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.bilateralFilter(gray, 5, 3, 10)
    canny_blurred = cv2.Canny(blurred, 30, 150)
    result['OpenCV'] = canny_blurred

    show_all_img(result)
