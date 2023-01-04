import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


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

def Grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def MedianBlur(image):
    image = cv2.medianBlur(image, 3, 0)
    return image

def Morphology_filter(image):
    print('Morphology Filtering...')
    '''
    A3 = 0.1*( (F open B1)close B2) + 0.9*( (F close B2) open B1)
    '''
    B1 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], np.uint8)

    B2 = np.array([[0, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0]], np.uint8)
    
    # opening
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, B1, iterations=1)
    # closing
    A1 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, B2, iterations=1)
    # opening
    A2 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, B2, iterations=1)
    # closing
    A2 = cv2.morphologyEx(A2, cv2.MORPH_OPEN, B1, iterations=1)

    A3 = 0.1*A1 + 0.9*A2
    return A3


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

def Imp_SobelFilter(image):

    image = Morphology_filter(Grayscale(image))
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    G_45 = np.zeros(image.shape)
    G_135 = np.zeros(image.shape)

    size = image.shape
    kernel_x = np.array(([-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]))

    kernel_y = np.array(([-1, -2, -1],
                         [0, 0, 0], 
                         [1, 2, 1]))
    
    kernel_45 = np.array(([-2, -1, 0],
                         [-1, 0, 1], 
                         [0, 1, 2]))

    kernel_135 = np.array(([0, 1, 2],
                         [-1, 0, 1], 
                         [-2, -1, 0]))

    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            target_image = image[i - 1 : i + 2, j - 1 : j + 2]
            G_x[i, j] = np.sum(np.multiply(target_image, kernel_x))
            G_y[i, j] = np.sum(np.multiply(target_image, kernel_y))
            G_45[i, j] = np.sum(np.multiply(target_image, kernel_45))
            G_135[i, j] = np.sum(np.multiply(target_image, kernel_135))
    
    convolved = np.sqrt(np.square(G_x) + np.square(G_y) + np.square(G_45) + np.square(G_135))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(np.square(G_y), np.square(G_x)))
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

def Otsu_threshold(image, lowrate = 0.1):
    print('Otsu thresholding...')
    # Otsu's thresholding
    image = image.astype("uint8")
    ret, th = cv2.threshold(image, 0, 255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    l_threshold = ret * lowrate
    h_threshold = ret

    return h_threshold, l_threshold

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


def Canny(image, low = 0, high = 30):
    image, angles = SobelFilter(image, method='gaussian')
    image = non_maximum_suppression(image, angles)
    gradient = np.copy(image)
    image = double_threshold_hysteresis(image, low, high)
    return image, gradient

def Imp_Canny(image):
    image, angles = Imp_SobelFilter(image)
    image = non_maximum_suppression(image, angles)
    gradient = np.copy(image)
    high, low = Otsu_threshold(image)
    image = double_threshold_hysteresis(image, low, high)
    return image, gradient

if __name__ == "__main__":

    input_path = ('jet.png')
    #output_path = ('jet_2.png')
    image = cv2.imread(input_path)
    image2, gradient2 = Imp_Canny(image)
    image1, gradient1 = Canny(image)

    result = {}
    result['Original'] = image
    result['Original_Canny'] = image1
    result['Improved_Canny'] = image2

    show_all_img(result)

    #cv2.imwrite(output_path, image)
