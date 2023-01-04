

from copy import copy
from pickletools import uint8
import cv2
import argparse
import random
import numpy as geek
import skimage.util.noise as noise
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import ndimage


#Resize
def img_resize(image, resize_height, resize_width):

    image_shape=np.shape(image)
    height=image_shape[0]
    width=image_shape[1]

    if (resize_height is None) and (resize_width is None):
        return image

    #Resize height
    if resize_height is not None:
        resize_width=int(width* (resize_height/height) )
    #Resize width
    elif resize_width is not None:
        resize_height=int(height* (resize_width/width) )
        
    img = cv2.resize(image, dsize=(resize_width, resize_height))

    return img

#RGB -> Gray
def RGB2GRAY(image):
    return cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)


#Padding
def padding(image , kernel_size):

    padding_size = (kernel_size[0] // 2, kernel_size[1] // 2)
    height = image.shape[0]
    width = image.shape[1]
    output_h = height + padding_size[0] * 2
    output_w = width + padding_size[1] * 2
    output = np.zeros( (output_h , output_w), dtype="uint8")

    for i in range(height):
        for j in range(width):
            output[i + padding_size[0]][j + padding_size[1]] = image[i][j] 

    return output

#Convolution
def convolution(image, kernel):

    kernel_size = kernel.shape
    output = np.zeros(image.shape, dtype = "uint8")
    pad_img = padding(image , kernel_size)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp = []
            result = 0
            temp.append( pad_img[ i:i+kernel_size[0], j:j+kernel_size[1] ] )
            image_data = np.row_stack(temp).flatten()
            kernel_data = np.row_stack(kernel).flatten()
            
            #Calculate
            for k in range(len(kernel_data)):
                result += image_data[k] * kernel_data[k]
            output[i][j] = result
    
    return output

#Sobel Filter
def sobel_filter(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    gx = convolution(image, Kx)
    gy = convolution(image, Ky)
    
    G = np.hypot(gx, gy)
    G = G / G.max() * 255
    theta = np.arctan2(gx, gy)
    
    return (G, theta)


#Smoothing
#Average
def box_filter(image, kernel_size = (3, 3)):

    all = kernel_size[0]* kernel_size[1]
    kernel = np.array([1]*all).reshape(kernel_size)

    avg_kernel = (1/ all)*kernel

    if len(image.shape) == 2:
        output = convolution(image, avg_kernel)
    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = convolution(image_r , avg_kernel)
        output_g = convolution(image_g , avg_kernel)
        output_b = convolution(image_b , avg_kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

#Gaussian Filter
def gs_filter(image, kernel_size = (3, 3)):
    
    #Gassian filter
    x1 = 1 - math.ceil(kernel_size[0] / 2)
    x2 = math.ceil(kernel_size[0] / 2)
    y1 = 1 - math.ceil(kernel_size[1] / 2)
    y2 = math.ceil(kernel_size[1] / 2)
    
    x, y = np.mgrid[x1:x2, y1:y2]

    gaussian_kernel = np.exp(-(x**2+y**2))

    #Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    if len(image.shape) == 2:
        output = convolution(image, gaussian_kernel)
    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = convolution(image_r , gaussian_kernel)
        output_g = convolution(image_g , gaussian_kernel)
        output_b = convolution(image_b , gaussian_kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

    
#Median filter
def median_img_filter(image, kernel_size = (3, 3)):

    def median_filter(image, kernel_size):
        height = image.shape[0]
        width = image.shape[1]
        output = np.zeros((height, width), dtype = "uint8")

        #Padding
        padding_image = padding(image , kernel_size)

        #Find Median
        for i in range(height):
            for j in range(width):
                temp = []
                temp.append( padding_image[ i:i+kernel_size[0], j:j+kernel_size[1] ] )
                #Median
                median = np.median(np.row_stack(temp).flatten())
                output[i][j] = median
        return output

    if len(image.shape) == 2:
        output = median_filter(image, kernel_size)
    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = median_filter(image_r , kernel_size )
        output_g = median_filter(image_g , kernel_size )
        output_b = median_filter(image_b , kernel_size )
        output = np.dstack((output_r, output_g, output_b))

    return output

# Histogram Equalization
def hist_equa_img(image):

    def hist_equa(image):
        data = np.zeros(256).astype(np.int64)
        image_f = image.flatten()
        for i in image_f:
            data[int(i)] += 1

        p = data/image_f.size
        p_sum = geek.cumsum(p)
        equal = np.around(p_sum * 255).astype('uint8')
        output = equal[image_f].reshape(image.shape)

        return equal[image]

    if len(image.shape) == 2:
        output = hist_equa(image)
    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = hist_equa(image_r )
        output_g = hist_equa(image_g )
        output_b = hist_equa(image_b )
        output = np.dstack((output_r, output_g, output_b))
        
        return output 

#Bilateral Filter
def bilatfilt(I, w, sd, sr):
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

#Show Image
def show_all_img(result, columns = None, rows = 1):

    if columns is None:
        columns = len(result)
    fig = plt.figure(figsize=(20, 20))

    for i, img_n in enumerate(result):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.title.set_text(img_n)
        # ax.set_title(i)  
        img = result[img_n]
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
    #plt.tight_layout()
    plt.show()