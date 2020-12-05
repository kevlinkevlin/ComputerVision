# %%
import numpy as np
import cv2
from tqdm import tqdm
import random

class Kernel:
    def __init__(self, init_list, origin):
        self.pattern = init_list
        self.origin = origin

    def get_directions(self):
        tmp_list = []
        for i in tqdm(range(len(self.pattern))):
            for j in range(len(self.pattern[0])):
                if self.pattern[i][j] == 1:
                    direction = (i - self.origin[0], j - self.origin[1])
                    tmp_list.append(direction)
        return tmp_list


img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)


octo_kernel_pattern = [
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]
]

L_shape_kernel_pattern = [
    [1, 1],
    [0, 1]
]
box_filter_3_3 = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]
box_filter_5_5 = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]

def GetGaussianNoise_Image(src,amplitude):
    gaussianNoise_Image = src.copy()
    for c in range(len(src)):
        for r in range(len(src[0])):
            noisePixel = int(src[c][r]) + amplitude * random.gauss(0,1)
            if noisePixel > 255:
                noisePixel = 255
            gaussianNoise_Image[c][r] = noisePixel
    return gaussianNoise_Image

def GetSaltAndPepper_Image(src,threshold):
    SaltAndPepper_Image = src.copy()
    for c in range(len(src)):
        for r in range(len(src[0])):
            random_Value = random.uniform(0,1)
            if(random_Value <= threshold):
                SaltAndPepper_Image[c][r] = 0
            elif (random_Value >= (1 - threshold)):
                SaltAndPepper_Image[c][r] = 255
    return SaltAndPepper_Image
# (a) Dilation
def dilation(src, kernel, directions):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            old = src[i][j]
            max_pixel = 0
            for direction in directions:
                new_i = i - direction[0]
                new_j = j - direction[1]

                if new_i >= 0 and new_i < len(src) and new_j >= 0 and new_j < len(src[0]):
                    tmp = src[new_i][new_j] + kernel.pattern[kernel.origin[0] +
                                                             direction[0]][kernel.origin[1] + direction[1]]
                    if tmp > 255:
                        tmp = 255
                    if tmp > max_pixel:
                        max_pixel = tmp
            new_img[i][j] = max_pixel

    return new_img

# (b) Erosion


def erosion(src, kernel, directions):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            old = src[i][j]
            min_pixel = 255
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]

                if new_i >= 0 and new_i < len(src) and new_j >= 0 and new_j < len(src[0]):
                    tmp = src[new_i][new_j] - kernel.pattern[kernel.origin[0] +
                                                             direction[0]][kernel.origin[1] + direction[1]]
                    if tmp < 0:
                        tmp = 0
                    if tmp < min_pixel:
                        min_pixel = tmp
            new_img[i][j] = min_pixel

    return new_img

# (c) Opening


def opening(src, kernel, directions):
    return dilation(erosion(src, kernel, directions), kernel, directions)
# (d) Closing


def closing(src, kernel, directions):
    return erosion(dilation(src, kernel, directions), kernel, directions)

# (e) Hit-and-miss transform


def hit_and_miss(src, J_kernel, J_directions, K_kernel, K_directions):
    img_J = erosion(src, J_kernel, J_directions)
    img_K = erosion(complement(src), K_kernel, K_directions)

    return union(src, img_J, img_K)


def union(src, J, K):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            if J[i][j] == 255 and K[i][j] == 255:
                new_img[i][j] = 255

    return new_img


def complement(src):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            if src[i][j] == 0:
                new_img[i][j] = 255
    return new_img
def box_filter(src,filter_direction):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            average,count = 0,0
            for direction in filter_direction:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if new_i >= 0 and new_i < len(src) and new_j >= 0 and new_j < len(src[0]):
                    average += src[new_i][new_j]
                    count += 1
            if count > 0:
                new_img[i][j] = average/count
    return new_img
def median_filter(src,filter_direction):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            sort_array,count = [],0
            for direction in filter_direction:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if new_i >= 0 and new_i < len(src) and new_j >= 0 and new_j < len(src[0]):
                    sort_array.append(src[new_i][new_j])
                    count += 1
            sort_array.sort()
            mid = (count + 1)/2 if ((count + 1)%2 == 0) else (count + 1)/2 + 1
            new_img[i][j] = sort_array[int(mid)-1]
    return new_img

octo_kernel = Kernel(octo_kernel_pattern, (2, 2))
octo_directions = octo_kernel.get_directions()

boxfilter_3_3 = Kernel(box_filter_3_3, (1, 1))
boxfilter_3_3_dir = boxfilter_3_3.get_directions()
boxfilter_5_5 = Kernel(box_filter_5_5, (2, 2))
boxfilter_5_5_dir = boxfilter_5_5.get_directions()

GaussianNoise_Image_10 = GetGaussianNoise_Image(img,10)
GaussianNoise_Image_10_box_3x3 = box_filter(GaussianNoise_Image_10,boxfilter_3_3_dir)
GaussianNoise_Image_10_median_3x3 = median_filter(GaussianNoise_Image_10,boxfilter_3_3_dir)
GaussianNoise_Image_10_box_5x5 = box_filter(GaussianNoise_Image_10,boxfilter_5_5_dir)
GaussianNoise_Image_10_median_5x5 = median_filter(GaussianNoise_Image_10,boxfilter_5_5_dir)
GaussianNoise_Image_10_open_to_close = closing(opening(GaussianNoise_Image_10,octo_kernel,octo_directions),octo_kernel,octo_directions)
GaussianNoise_Image_10_close_to_open = opening(closing(GaussianNoise_Image_10,octo_kernel,octo_directions),octo_kernel,octo_directions)

GaussianNoise_Image_30 = GetGaussianNoise_Image(img,30)
GaussianNoise_Image_30_box_3x3 = box_filter(GaussianNoise_Image_30,boxfilter_3_3_dir)
GaussianNoise_Image_30_median_3x3 = median_filter(GaussianNoise_Image_30,boxfilter_3_3_dir)
GaussianNoise_Image_30_box_5x5 = box_filter(GaussianNoise_Image_30,boxfilter_5_5_dir)
GaussianNoise_Image_30_median_5x5 = median_filter(GaussianNoise_Image_30,boxfilter_5_5_dir)
GaussianNoise_Image_30_open_to_close = closing(opening(GaussianNoise_Image_30,octo_kernel,octo_directions),octo_kernel,octo_directions)
GaussianNoise_Image_30_close_to_open = opening(closing(GaussianNoise_Image_30,octo_kernel,octo_directions),octo_kernel,octo_directions)

SaltAndPepper_Image_0_05 = GetSaltAndPepper_Image(img,0.05)
SaltAndPepper_Image_0_05_box_3x3 = box_filter(SaltAndPepper_Image_0_05,boxfilter_3_3_dir)
SaltAndPepper_Image_0_05_median_3x3 = median_filter(SaltAndPepper_Image_0_05,boxfilter_3_3_dir)
SaltAndPepper_Image_0_05_box_5x5 = box_filter(SaltAndPepper_Image_0_05,boxfilter_5_5_dir)
SaltAndPepper_Image_0_05_median_5x5 = median_filter(SaltAndPepper_Image_0_05,boxfilter_5_5_dir)
SaltAndPepper_Image_0_05_open_to_close = closing(opening(SaltAndPepper_Image_0_05,octo_kernel,octo_directions),octo_kernel,octo_directions)
SaltAndPepper_Image_0_05_close_to_open = opening(closing(SaltAndPepper_Image_0_05,octo_kernel,octo_directions),octo_kernel,octo_directions)

SaltAndPepper_Image_0_1 = GetSaltAndPepper_Image(img,0.1)
SaltAndPepper_Image_0_1_box_3x3 = box_filter(SaltAndPepper_Image_0_1,boxfilter_3_3_dir)
SaltAndPepper_Image_0_1_median_3x3 = median_filter(SaltAndPepper_Image_0_1,boxfilter_3_3_dir)
SaltAndPepper_Image_0_1_box_5x5 = box_filter(SaltAndPepper_Image_0_1,boxfilter_5_5_dir)
SaltAndPepper_Image_0_1_median_5x5 = median_filter(SaltAndPepper_Image_0_1,boxfilter_5_5_dir)
SaltAndPepper_Image_0_1_open_to_close = closing(opening(SaltAndPepper_Image_0_1,octo_kernel,octo_directions),octo_kernel,octo_directions)
SaltAndPepper_Image_0_1_close_to_open = opening(closing(SaltAndPepper_Image_0_1,octo_kernel,octo_directions),octo_kernel,octo_directions)

cv2.imwrite('GaussianNoise_Image_10.bmp', GaussianNoise_Image_10)
cv2.imwrite('GaussianNoise_Image_10_box_3x3.bmp', GaussianNoise_Image_10_box_3x3)
cv2.imwrite('GaussianNoise_Image_10_median_3x3.bmp', GaussianNoise_Image_10_median_3x3)
cv2.imwrite('GaussianNoise_Image_10_box_5x5.bmp', GaussianNoise_Image_10_box_5x5)
cv2.imwrite('GaussianNoise_Image_10_median_5x5.bmp', GaussianNoise_Image_10_median_5x5)
cv2.imwrite('GaussianNoise_Image_10_open_to_close.bmp', GaussianNoise_Image_10_open_to_close)
cv2.imwrite('GaussianNoise_Image_10_close_to_open.bmp', GaussianNoise_Image_10_close_to_open)



cv2.imwrite('GaussianNoise_Image_30.bmp', GaussianNoise_Image_30)
cv2.imwrite('GaussianNoise_Image_30_box_3x3.bmp', GaussianNoise_Image_30_box_3x3)
cv2.imwrite('GaussianNoise_Image_30_median_3x3.bmp', GaussianNoise_Image_30_median_3x3)
cv2.imwrite('GaussianNoise_Image_30_box_5x5.bmp', GaussianNoise_Image_30_box_5x5)
cv2.imwrite('GaussianNoise_Image_30_median_5x5.bmp', GaussianNoise_Image_30_median_5x5)
cv2.imwrite('GaussianNoise_Image_30_open_to_close.bmp', GaussianNoise_Image_30_open_to_close)
cv2.imwrite('GaussianNoise_Image_30_close_to_open.bmp', GaussianNoise_Image_30_close_to_open)



cv2.imwrite('SaltAndPepper_Image_0_05.bmp', SaltAndPepper_Image_0_05)
cv2.imwrite('SaltAndPepper_Image_0_05_box_3x3.bmp', SaltAndPepper_Image_0_05_box_3x3)
cv2.imwrite('SaltAndPepper_Image_0_05_median_3x3.bmp', SaltAndPepper_Image_0_05_median_3x3)
cv2.imwrite('SaltAndPepper_Image_0_05_box_5x5.bmp', SaltAndPepper_Image_0_05_box_5x5)
cv2.imwrite('SaltAndPepper_Image_0_05_median_5x5.bmp', SaltAndPepper_Image_0_05_median_5x5)
cv2.imwrite('SaltAndPepper_Image_0_05_open_to_close.bmp', SaltAndPepper_Image_0_05_open_to_close)
cv2.imwrite('SaltAndPepper_Image_0_05_close_to_open.bmp', SaltAndPepper_Image_0_05_close_to_open)


cv2.imwrite('SaltAndPepper_Image_0_1.bmp', SaltAndPepper_Image_0_1)
cv2.imwrite('SaltAndPepper_Image_0_1_box_3x3.bmp', SaltAndPepper_Image_0_1_box_3x3)
cv2.imwrite('SaltAndPepper_Image_0_1_median_3x3.bmp', SaltAndPepper_Image_0_1_median_3x3)
cv2.imwrite('SaltAndPepper_Image_0_1_box_5x5.bmp', SaltAndPepper_Image_0_1_box_5x5)
cv2.imwrite('SaltAndPepper_Image_0_1_median_5x5.bmp', SaltAndPepper_Image_0_1_median_5x5)
cv2.imwrite('SaltAndPepper_Image_0_1_open_to_close.bmp', SaltAndPepper_Image_0_1_open_to_close)
cv2.imwrite('SaltAndPepper_Image_0_1_close_to_open.bmp', SaltAndPepper_Image_0_1_close_to_open)


cv2.waitKey(0)
# %%
