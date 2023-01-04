import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

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

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    

def Grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def MedianBlur(image):
    image = cv2.medianBlur(image, 3, 0)
    return image

def SobelFilter(image, method = 'median'):
    if method == 'median':
        image = Grayscale(MedianBlur(image))
    else:
        image = Grayscale(GaussianBlur(image))
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))

    '''
    The gradient magnitude and gradient direction of the central pixel are calculated as follows:
    Gxy = [[a0, a3, a6],
          [a1, a4, a7],
          [a2, a5, a8]]
    px=(a6+2a7+a8)−(a0+2a3+a6)
    py=(a6+2a7+a8)−(a0+a1+a2)
    '''

    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            target_image = image[i - 1 : i + 2, j - 1 : j + 2]
            G_x[i, j] = np.sum(np.multiply(target_image, kernel_x))
            G_y[i, j] = np.sum(np.multiply(target_image, kernel_y))
    
    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    '''
    The gradient direction is calculated as follows:
    G1(i,j)=root(p2x+p2y) 2:square
    theta(i,j) = arctan(px/py)
    '''
    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

def adaptive_threshold(image):
    '''
    generate the gradient amplitude histogram as G1
    '''
    print('Adaptive Choosing Thresholds...')
    # create the histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    G1 = histogram

    '''
    Select the first zero of the amplitude as a high threshold, and the low value take 0.4 times of this high threshold
    '''
    i = 0
    while i <= 255 :
        #print(i)
        i +=1
        if G1[i+1] - G1[i] == 0:
            break
    
    h_threshold = G1[i]
    l_threshold = h_threshold*0.4

    return h_threshold, l_threshold

        
def double_threshold_hysteresis(image, low, high):
    print('Double Thresholding...')
    weak = 0
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = np.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = np.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape
    
    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y]  == weak)):
                result[new_x, new_y] = strong
                np.append(strong_x, new_x)
                np.append(strong_y, new_y)
    result[result != strong] = 0
    return result

def Filtering_Generalized_Chains(image, low, high):
    print('Filtering Generalized Chains...')
    weak = 0
    strong = 255
    size = image.shape
    result = np.zeros(size)
    weak_x, weak_y = np.where((image > low) & (image <= high))
    strong_x, strong_y = np.where(image >= high)
    grad_avg = np.average(image)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak

    '''
    Select strong edge points as the edge of the starting point and link them with weak points to form edge chains, 
    and calculate the average of edge chains by (8), 
    then remove the generalized edge chains, which are smaller than the average of the gradient maximum of the image.
    '''
    strong_points_list = []
    for i in range(len(strong_x)):
        point = (strong_x[i], strong_y[i])
        strong_points_list.append(point)

    weak_points_list = []
    for i in range(len(weak_x)):
        point = (weak_x[i], weak_y[i])
        weak_points_list.append(point)

    # arr = np.array(distance_list)
    # dmax = np.max(arr)
    # dmin = np.min(arr)
    # davg = (dmax + dmin)/2

    size = image.shape
    distance_list = []
    progress = tqdm(total=len(strong_x))
    while len(strong_x):
        progress.update(1)
        local_distance_list = []
        x = strong_x[0]
        y = strong_y[0]
        strong_x = np.delete(strong_x, 0)
        strong_y = np.delete(strong_y, 0)
        for i in range(len(weak_x)):
            d = abs(image[x,y] - image[weak_x[i], weak_y[i]])
            local_distance_list.append(d)
            avg_local_dis = np.average(np.array(local_distance_list))
            # distance_list.append(avg_local_dis)

        local_dmax = np.max(np.array(local_distance_list))
        local_dmin = np.min(np.array(local_distance_list))
        if avg_local_dis > (local_dmax + local_dmin)/2:
            if image[x,y] < avg_local_dis:
                result[x,y] = 0

        # for p in weak_points_list:
        #    d = distance((x,y), p)
        #    local_distance_list.append(d)
        # avg_local_dis = np.average(np.array(local_distance_list))
        # distance_list.append(avg_local_dis)

    # d_avg = np.average(np.array(distance_list))
    # p_strong_x, p_strong_y = np.where(image < d_avg)
    # result[p_strong_x, p_strong_y] = 0

    # result[result != strong] = 0
    return result, avg_local_dis

def Canny(image, low, high):
    image, angles = SobelFilter(image, method='gaussian')
    image = non_maximum_suppression(image, angles)
    gradient = np.copy(image)
    image = double_threshold_hysteresis(image, low, high)
    return image, gradient

def Imp_Canny(image):
    image, angles = SobelFilter(image, method='median')
    image = non_maximum_suppression(image, angles)
    gradient = np.copy(image)
    high, low = adaptive_threshold(image)
    image, average = Filtering_Generalized_Chains(image, low, high)
    return image, gradient, average

if __name__ == "__main__":

    input_path = ('jet.png')
    #output_path = ('jet_2.png')
    image = cv2.imread(input_path)
    image1, gradient1 = Canny(image, low = 0, high = 25)
    image2, gradient2, average = Imp_Canny(image)
    print('average: ', average)

    result = {}
    result['Original'] = image
    result['Original_Canny'] = image1
    result['Improved_Canny'] = image2

    show_all_img(result)
