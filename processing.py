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

# RGB -> HSV
def RGB2HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    

#Noise
#Salt and pepper Noise
def impulse_noise(image, prob = 0.5):
    output = image.copy()
    h, w = image.shape[:2] #height, weight
    ps = 1 - prob

    for i in range(h):
        for j in range(w):
            rand_prob = random.random()
            if rand_prob > 0 and rand_prob <= prob/2:   #rdn < prob
                output[i, j] = 0
            elif rand_prob > prob/2 and rand_prob <= prob:    #rand_prob > ps
                output[i, j] = 255
            else:
                output[i, j] = image[i, j] 
    return output

#Noise
#Gaussian Noise
def gs_noise(image, mean=0, sigma=0.1):

    # Standardize
    output = image / 255.0
    # gs_noise
    noise = np.random.normal(mean, sigma, image.shape)
    output = output + noise
    # clip 0~1
    output = np.clip(output, 0, 1)
    # float -> int (0~1 -> 0~255)
    output = np.uint8(output*255)

    return output

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
def conv(image, kernel):

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


# Sharpen
def Laplacian_filter(image):

    kernel = np.array(
        [[0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]])

    if len(image.shape) == 2:
        output = conv(image, kernel)
    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = conv(image_r , kernel)
        output_g = conv(image_g , kernel)
        output_b = conv(image_b , kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

# Sharpen
def sharpen_filter(image):

    kernel = np.array(
        [[-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])
    
    if len(image.shape) == 2:
        output = conv(image,kernel)
    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = conv(image_r , kernel)
        output_g = conv(image_g , kernel)
        output_b = conv(image_b , kernel)
        output = np.dstack((output_r, output_g, output_b))

    return output

# Sharpen
def unsharp_masking(image, kernel_size = (3, 3), k = 1):

    mask = image - box_filter(np.copy(image), kernel_size) 
    output = image + k*mask

    return output

#Smoothing
#Average
def box_filter(image, kernel_size = (3, 3)):

    all = kernel_size[0]* kernel_size[1]
    kernel = np.array([1]*all).reshape(kernel_size)

    avg_kernel = (1/ all)*kernel

    if len(image.shape) == 2:
        output = conv(image, avg_kernel)
    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = conv(image_r , avg_kernel)
        output_g = conv(image_g , avg_kernel)
        output_b = conv(image_b , avg_kernel)
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
        output = conv(image, gaussian_kernel)
    else :
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = conv(image_r , gaussian_kernel)
        output_g = conv(image_g , gaussian_kernel)
        output_b = conv(image_b , gaussian_kernel)
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

#Max filter
def max_img_filter(image , kernel_size = (3, 3)):

    def max_filter(image , kernel_size):

        image = np.copy(image)
        o_height = image.shape[0]
        o_width = image.shape[1]

        #Padding
        image = padding(image , kernel_size)

        #Max
        result = np.zeros((o_height, o_width), dtype=np.uint8) 
        for i in range(image.shape[0] - kernel_size[0]+1):
            for j in range(image.shape[1] - kernel_size[1]+1):
                result[i, j] = np.max(image[ i:i+kernel_size[0], j:j+kernel_size[1] ]) 

        return result

    if len(image.shape) == 2:
        output = max_filter(image, kernel_size)
    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = max_filter(image_r , kernel_size )
        output_g = max_filter(image_g , kernel_size )
        output_b = max_filter(image_b , kernel_size )
        output = np.dstack((output_r, output_g, output_b))

    return output

#Min filter
def min_img_filter(image , kernel_size = (3, 3)):

    def min_filter(image , kernel_size):

        image = np.copy(image)
        o_height = image.shape[0]
        o_width = image.shape[1]

        #Padding
        image = padding(image , kernel_size)

        #Min
        result = np.zeros((o_height, o_width), dtype=np.uint8) 
        for i in range(image.shape[0] - kernel_size[0]+1):
            for j in range(image.shape[1] - kernel_size[1]+1):
                result[i, j] = np.min(image[ i:i+kernel_size[0], j:j+kernel_size[1] ]) 

        return result

    if len(image.shape) == 2:
        output = min_filter(image, kernel_size)
    else:
        image_r = image[:, :, 0]
        image_g = image[:, :, 1]
        image_b = image[:, :, 2]

        output_r = min_filter(image_r , kernel_size )
        output_g = min_filter(image_g , kernel_size )
        output_b = min_filter(image_b , kernel_size )
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
  

# Gamma correction
def gamma_correction(image, r, c=1):
    output = image.copy()
    output = output/ 255
    output = (1/c * output) ** r

    output *= 255
    output = output.astype(np.uint8)

    return output

#RGB Gamma correction   
def RGB_gamma_correction(image, r, channel):

    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]

    if channel == 'R':
        output_r = gamma_correction(image_r, r)
        output = np.dstack((output_r, image_g, image_b))
    elif channel == 'G':
        output_g = gamma_correction(image_g, r)
        output = np.dstack((image_r, output_g, image_b))
    else:
        output_b = gamma_correction(image_b, r)
        output = np.dstack((image_r, image_g, output_b))

    return output


# Alpha and Beta correction
def alpha_beta_correction(image, a, b):
    output = np.zeros(image.shape, image.dtype)

    # Initialize values
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                output[y,x,c] = np.clip(a*image[y,x,c] + b, 0, 255)

    return output
            

def reduce_highlights(img, criteria = 200, alpha = 0.1, beta = 0.1):

    image_r = img[:, :, 0]
    image_g = img[:, :, 1]
    image_b = img[:, :, 2]

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(image_b, criteria, 255, 0)  
    contours, hierarchy  = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8) 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)   
        img_zero[y:y+h, x:x+w] = 255 
        mask = img_zero 

    result = cv2.illuminationChange(img, mask, alpha=alpha, beta=beta) 
        
    return result



def adaptive_median_filter(image, max_size: int=7): 

    def zero_padding(src, padding_left: int, padding_right: int, padding_top: int, padding_bottom: int): 
        
        height, width = src.shape 
        # Vertical Zero padding for source 
        boundary_top = np.zeros((padding_top, width)) 
        boundary_bottom = np.zeros((padding_bottom, width)) 
        result = np.vstack((boundary_top, src, boundary_bottom)) 
        # Horizontal Zero padding for source 
        boundary_left = np.zeros((height+padding_top+padding_bottom, padding_left)) 
        boundary_right = np.zeros((height+padding_top+padding_bottom, padding_right)) 
        result = np.hstack((boundary_left, result, boundary_right)) 
        
        return result

    # For Gray image
    assert max_size % 2 ==1, 'kernel size must be odd.' 
    image = np.copy(image) 
    height, width = image.shape 
    kernel_h, kernel_w = (max_size-1)//2, (max_size-1)//2 

    image = zero_padding(image, kernel_w, kernel_w, kernel_h, kernel_h) 

    #Padding
    # kernel_size = (kernel_h, kernel_w)
    # image = padding(image , kernel_size)

    filter_size = 1 
    result = np.zeros(image.shape, dtype=np.uint8) 
    for i in range(kernel_h, height+kernel_h): 
        for j in range(kernel_w, width+kernel_w): 
            filter_size = 1 
            while filter_size <= kernel_w: 
                local_med = np.median(image[i-filter_size:i+filter_size+1, j-filter_size:j+filter_size+1]) 
                local_max = np.max(image[i-filter_size:i+filter_size+1, j-filter_size:j+filter_size+1])
                local_min = np.min(image[i-filter_size:i+filter_size+1, j-filter_size:j+filter_size+1]) 
                
                if local_med==local_max or local_med==local_min: 
                    result[i, j] = local_med 
                    filter_size += 1 
                elif image[i, j]==local_max or image[i, j]==local_min: 
                        result[i, j] = local_med 
                        break 
                else: 
                    result[i, j] = image[i, j] 
                    break 
                
    result = result[kernel_h:height+kernel_h, kernel_w:width+kernel_w] 
    return result


