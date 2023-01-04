from concurrent.futures.process import _MAX_WINDOWS_WORKERS
from re import I
from PIL import Image
from PIL.Image import core as _imaging
import os
import sys
import time
import numpy as np
import cv2
import sys
import math
from copy import copy

import random
import skimage.util.noise as noise
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

from os import listdir
from os.path import isfile, join
import numpy
from processing import*

def show_img(img):
    plt.figure(figsize=(15,15)) 
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()

# def show_all_img(result, columns = None, rows = 1):

#     if columns is None:
#         columns = len(result)
#     fig = plt.figure(figsize=(20, 20))

#     # fig.suptitle('Median Filter with different kernel sizes')
#     # plt.xlabel('Median Filter', fontweight='bold')
#     for i, img_n in enumerate(result):
#         ax = fig.add_subplot(rows, columns, i+1)
#         ax.title.set_text(img_n)
#         # ax.set_title(i)  
#         img = result[img_n]
#         if len(img.shape) == 2:
#             plt.imshow(img, cmap='gray')
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             plt.imshow(img)
#     plt.show()

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
    plt.show()


#All files
img_file = 'example'
onlyfiles = [f for f in listdir(img_file) if isfile(join(img_file, f))]
print(onlyfiles)   

# Read Image
img1 = cv2.imread(os.path.join(img_file, onlyfiles[0]))
img2 = cv2.imread(os.path.join(img_file, onlyfiles[1]))
img3 = cv2.imread(os.path.join(img_file, onlyfiles[2]))
img4 = cv2.imread(os.path.join(img_file, onlyfiles[3]))
img5 = cv2.imread(os.path.join(img_file, onlyfiles[4]))
img6 = cv2.imread(os.path.join(img_file, onlyfiles[5]))

#Image 1 - Newspaper woman
input = img5
print(input.shape)
input = img_resize(input, None, 512)

# output
o_input = input
g_input = RGB2GRAY(input)

#Processing
result  = {}
result['Original'] = g_input

#test 0
filters = [box_filter, median_img_filter]
k_sizes = (7, 9, 11)

for filter in filters:
    for k_size in k_sizes:
        kernel_size = (k_size, k_size)
        print('filter:', filter, ' kernel size: ', kernel_size)
        img_sm = filter(g_input, kernel_size=kernel_size)
        filter_name = str(filter).split()[1]
        name = 'Kernel size:{}x{}, Filter:{}'.format(k_size, k_size, filter_name)
        result[name] = img_sm

show_all_img(result, columns = 4, rows = 2)

#test 1
kernel_test = (3, 5, 7, 9, 11)
for k in kernel_test:
    kernel_size = (k, k)
    img_ft = median_img_filter(g_input, kernel_size=kernel_size)
    result['Kernel size:{}x{}'.format(k, k)] = img_ft

show_all_img(result)

#test 2
result2 = {}
sharpen_filters = [unsharp_masking, sharpen_filter, Laplacian_filter]
img_ft = median_img_filter(g_input, kernel_size = (11, 11))
result2['Median Filter'] = img_ft
for m in sharpen_filters:
    img_sp = m(img_ft)
    result2[str(m).split()[1]] = img_sp

show_all_img(result2)

#test 3
result3 = {}
ks = [1, 10, 20]
k_sizes = [7, 9, 11]
img_ft = median_img_filter(g_input, kernel_size = (11, 11))
result3['Median Filter'] = img_ft
for k in ks:
    for k_size in k_sizes:
        kernel_size = (k_size, k_size)
        print('k:', k, ' kernel size: ', kernel_size)
        img_sp = unsharp_masking(img_ft, kernel_size=(kernel_size), k = k)
        name = 'Kernel size:{}x{}, k:{}'.format(k_size, k_size, k)
        result3[name] = img_sp

show_all_img(result3, 5, 2)

Final of P1
final_result = {}
img_ft = median_img_filter(g_input, kernel_size = (11, 11))
img_sp = unsharp_masking(img_ft, kernel_size=(7, 7), k = 1)
final_result['result'] = img_sp
show_all_img(final_result)

#################################################
#Image 2 - NCTU
input = img2
print(input.shape)
input = img_resize(input, None, 1024)

# output
o_input = input
# convert img to gray
g_input = RGB2GRAY(input)

#Processing
result  = {}
result['Original'] = o_input

#test 1
r = 0.5
model = 'HSV'
hue, sat, val = cv2.split(RGB2HSV(image=o_input))
val_gamma = gamma_correction(val, 0.5)
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_r = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
result2['Gamma:{}, Color Model:{}'.format(r, model)] = img_r

alpha = [1, 2, 3]
beta = [25, 50, 75]
for a in alpha:
    for b in beta:
        print('Alpha: ', a, 'Beta: ', b)
        img_ab = alpha_beta_correction(o_input, a, b)
        result['Alpha:{}, Beta:{}'.format(a, b)] = img_ab

show_all_img(result, 5, 2)

#test 2  Gamma, Color model
result2 = {}
#alpha 3, beta 25
#r = 0.67, model = HSV
a = 3
b = 25
img_ab = alpha_beta_correction(o_input, a, b)
result2["Alpha_Beta_Correction"] = img_ab
rs = [0.67, 0.5, 0.33]
models = ['RGB', 'HSV']

for r in rs:
    for model in models:
        if model == 'RGB':
            img_r = gamma_correction(img_ab, r)
            result2['Gamma:{}, Color Model:{}'.format(r, model)] = img_r
        else:
            hue, sat, val = cv2.split(RGB2HSV(image=img_ab))
            val_gamma = gamma_correction(val, r)
            hsv_gamma = cv2.merge([hue, sat, val_gamma])
            img_r = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)
            result2['Gamma:{}, Color Model:{}'.format(r, model)] = img_r

show_all_img(result2, 4, 2)

#test 3
result3 = {}
#alpha 3, beta 25
#r = 0.67, model = HSV
a = 3
b = 25
r = 0.67
img_ab = alpha_beta_correction(o_input, a, b)
hue, sat, val = cv2.split(RGB2HSV(image=img_ab))
val_gamma = gamma_correction(val, r)
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_r = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

img_br = RGB_gamma_correction(img_r, r = 1.5, channel = 'B')
img_light = reduce_highlights(img_br, criteria = 253, alpha=0.12, beta = 0.12)


#result
result3["Gamma_Correction"] = img_r
result3["Blue_Gamma_Correction"] = img_br
result3["Light_Correction"] = img_light

show_all_img(result3)

#Final of P1
final_result = {}
result3 = {}
#alpha 3, beta 25
#r = 0.67, model = HSV
a = 3
b = 25
r = 0.67
img_ab = alpha_beta_correction(o_input, a, b)
hue, sat, val = cv2.split(RGB2HSV(image=img_ab))
val_gamma = gamma_correction(val, r)
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_r = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

img_br = RGB_gamma_correction(img_r, r = 1.5, channel = 'B')
img_light = reduce_highlights(img_br, criteria = 253, alpha=0.12, beta = 0.12)

final_result['result'] = img_light
show_all_img(final_result)

#################################################
#Image 3 - 
input = img5
print(input.shape)
input = img_resize(input, None, 960)

# output
o_input = input
# convert img to gray
g_input = RGB2GRAY(input)

#Processing
result  = {}
result['Original'] = g_input

#test 1
noises = [gs_noise, impulse_noise]

for n in noises:
    if n == gs_noise:
        sigma = [0.1, 0.5, 0.9]
        for s in sigma:
            img_n = n(g_input, s)
            noise_name = str(n).split()[1]
            name = 'Noise:{}, Sigma:{}'.format(noise_name, s)
            result[name] = img_n
    else:
        prob = [0.1, 0.5, 0.9]
        for p in prob:
            img_n = n(g_input, p)
            noise_name = str(n).split()[1]
            name = 'Noise:{}, Probability:{}'.format(noise_name, p)
            result[name] = img_n

show_all_img(result, 4, 2)

#test 2
result2 = {}
img_n = impulse_noise(g_input, 0.1)
result2['Impulse Noise'] = img_n

filters = [max_img_filter, min_img_filter, median_img_filter, adaptive_median_filter]

for filter in filters:
    img_filter = filter(img_n)
    filter_name = str(filter).split()[1]
    name = 'Filter:{}'.format(filter_name)
    result2[name] = img_filter
    
show_all_img(result2, 5, 1)

#test 3
result3 = {}
img_n = impulse_noise(g_input, 0.3)
result3['Impulse Noise'] = img_n

kernel_size = [5, 7, 9]
   
#Median
for k in kernel_size:
    print('filter: Median Filter, kernel size: ', k)
    k_size = (k, k)
    img_ft = median_img_filter(img_n, k_size)
    name = 'Median Filter, Kernel size:{}x{}'.format(k, k)
    result3[name] = img_ft

#Adaptive Median
for k in kernel_size:
    print('Adaptive Median Filter, kernel size: ', k)
    k_size = 2*k+1
    img_ft = adaptive_median_filter(img_n, k_size)
    name = 'Adaptive Median Filter, Max Kernel size:{}x{}'.format(k, k)
    result3[name] = img_ft

show_all_img(result3, 4, 2)

#final
final_result = {}
img_n = impulse_noise(g_input, 0.3)
final_result['Impulse Noise'] = img_n

kernel_size = [5, 7, 9]
   
#Adaptive Median
k = 7
k_size = 2*k+1
img_ft = adaptive_median_filter(img_n, k_size)
name = 'Output'
final_result[name] = img_ft

show_all_img(final_result)

#################################################
#Image 4 - 
input = img3
print(input.shape)
input = img_resize(input, None, 900)

# output
o_input = input
# convert img to gray
g_input = RGB2GRAY(input)

#Processing
result  = {}
result['Original'] = o_input

#test 1 histogram
img_his = hist_equa_img(o_input)
result['Histogram Equalization'] = img_his

show_all_img(result)

#test 2
result2 = {}
result2['Histogram Equalization'] = img_his
filters = [gs_filter, box_filter, median_img_filter]
kernel_size = [3, 5, 7]

for filter in filters:
    for k in kernel_size:
        print(filter, k)
        k_size = (k, k)
        img_ft = filter(img_his, k_size)
        filter_name = str(filter).split()[1]
        name = 'Filter:{} Kernel size:{}x{}'.format(filter_name, k, k)
        result2[name] = img_ft

show_all_img(result2, 5, 2)

#test 3
result3 = {}
sharpen_filters = [unsharp_masking, sharpen_filter, Laplacian_filter]
img_ft = gs_filter(img_his)
result3['Smoothing'] = img_ft
for m in sharpen_filters:
    print(m)
    img_sp = m(img_ft)
    result3[str(m).split()[1]] = img_sp

show_all_img(result3)

#test 4
result4 = {}
#input
result4['Smoothing'] = img_ft
rs = [0.9, 0.7, 0.5]
for r in rs:
    img_br = RGB_gamma_correction(img_ft, r = r, channel = 'B')
    result4['Gamma in B:{}'.format(r)] = img_br
    
show_all_img(result4)

#Final
final_result = {}
img_his = hist_equa_img(o_input)
img_ft = gs_filter(img_his)
img_br = RGB_gamma_correction(img_ft, r = 0.7, channel = 'B')
final_result['output'] = img_br

show_all_img(final_result)