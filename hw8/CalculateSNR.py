#%%
import numpy as np
import cv2
import math
img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
img_size = len(img)*len(img[0])
Mean_s = 0
for i in range(len(img)):
    for j in range(len(img[0])):
        Mean_s += float(img[i][j])/255
Mean_s /= img_size
VS = 0
for i in range(len(img)):
    for j in range(len(img[0])):
        VS += np.square(float(img[i][j])/255 - Mean_s)
VS /= img_size

def Get_SNR(origin,src,mean_s,size):
    mean_noise = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            mean_noise += (float(src[i][j]/255) - float(origin[i][j])/255)
    mean_noise /= size
    VN = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            VN += np.square(float(src[i][j])/255 - float(origin[i][j])/255 - mean_noise)
    VN /= size
    SNR = math.log(math.sqrt(VS)/math.sqrt(VN),10)*20
    print(SNR)
    return SNR



GaussianNoise_Image_10 = cv2.imread("GaussianNoise_Image_10.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10.bmp")
Get_SNR(img,GaussianNoise_Image_10,Mean_s,img_size)

GaussianNoise_Image_10_box_3x3 = cv2.imread("GaussianNoise_Image_10_box_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10_box_3x3.bmp")
Get_SNR(img,GaussianNoise_Image_10_box_3x3,Mean_s,img_size)

GaussianNoise_Image_10_box_5x5 = cv2.imread("GaussianNoise_Image_10_box_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10_box_5x5.bmp")
Get_SNR(img,GaussianNoise_Image_10_box_5x5,Mean_s,img_size)

GaussianNoise_Image_10_median_3x3 = cv2.imread("GaussianNoise_Image_10_median_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10_median_3x3.bmp")
Get_SNR(img,GaussianNoise_Image_10_median_3x3,Mean_s,img_size)

GaussianNoise_Image_10_median_5x5 = cv2.imread("GaussianNoise_Image_10_median_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10_median_5x5.bmp")
Get_SNR(img,GaussianNoise_Image_10_median_5x5,Mean_s,img_size)

GaussianNoise_Image_10_close_to_open = cv2.imread("GaussianNoise_Image_10_close_to_open.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10_close_to_open.bmp")
Get_SNR(img,GaussianNoise_Image_10_close_to_open,Mean_s,img_size)

GaussianNoise_Image_10_open_to_close = cv2.imread("GaussianNoise_Image_10_open_to_close.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_10_open_to_close.bmp")
Get_SNR(img,GaussianNoise_Image_10_open_to_close,Mean_s,img_size)
   




GaussianNoise_Image_30 = cv2.imread("GaussianNoise_Image_30.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30.bmp")
Get_SNR(img,GaussianNoise_Image_30,Mean_s,img_size)

GaussianNoise_Image_30_box_3x3 = cv2.imread("GaussianNoise_Image_30_box_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30_box_3x3.bmp")
Get_SNR(img,GaussianNoise_Image_30_box_3x3,Mean_s,img_size)

GaussianNoise_Image_30_box_5x5 = cv2.imread("GaussianNoise_Image_30_box_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30_box_5x5.bmp")
Get_SNR(img,GaussianNoise_Image_30_box_5x5,Mean_s,img_size)

GaussianNoise_Image_30_median_3x3 = cv2.imread("GaussianNoise_Image_30_median_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30_median_3x3.bmp")
Get_SNR(img,GaussianNoise_Image_30_median_3x3,Mean_s,img_size)

GaussianNoise_Image_30_median_5x5 = cv2.imread("GaussianNoise_Image_30_median_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30_median_5x5.bmp")
Get_SNR(img,GaussianNoise_Image_30_median_5x5,Mean_s,img_size)

GaussianNoise_Image_30_close_to_open = cv2.imread("GaussianNoise_Image_30_close_to_open.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30_close_to_open.bmp")
Get_SNR(img,GaussianNoise_Image_30_close_to_open,Mean_s,img_size)

GaussianNoise_Image_30_open_to_close = cv2.imread("GaussianNoise_Image_30_open_to_close.bmp", cv2.IMREAD_GRAYSCALE)
print("GaussianNoise_Image_30_open_to_close.bmp")
Get_SNR(img,GaussianNoise_Image_30_open_to_close,Mean_s,img_size)


SaltAndPepper_Image_0_1 = cv2.imread("SaltAndPepper_Image_0_1.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1,Mean_s,img_size)

SaltAndPepper_Image_0_1_box_3x3 = cv2.imread("SaltAndPepper_Image_0_1_box_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1_box_3x3.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1_box_3x3,Mean_s,img_size)

SaltAndPepper_Image_0_1_box_5x5 = cv2.imread("SaltAndPepper_Image_0_1_box_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1_box_5x5.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1_box_5x5,Mean_s,img_size)

SaltAndPepper_Image_0_1_median_3x3 = cv2.imread("SaltAndPepper_Image_0_1_median_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1_median_3x3.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1_median_3x3,Mean_s,img_size)

SaltAndPepper_Image_0_1_median_5x5 = cv2.imread("SaltAndPepper_Image_0_1_median_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1_median_5x5.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1_median_5x5,Mean_s,img_size)

SaltAndPepper_Image_0_1_close_to_open = cv2.imread("SaltAndPepper_Image_0_1_close_to_open.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1_close_to_open.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1_close_to_open,Mean_s,img_size)

SaltAndPepper_Image_0_1_open_to_close = cv2.imread("SaltAndPepper_Image_0_1_open_to_close.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_1_open_to_close.bmp")
Get_SNR(img,SaltAndPepper_Image_0_1_open_to_close,Mean_s,img_size)


SaltAndPepper_Image_0_05 = cv2.imread("SaltAndPepper_Image_0_05.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05,Mean_s,img_size)

SaltAndPepper_Image_0_05_box_3x3 = cv2.imread("SaltAndPepper_Image_0_05_box_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05_box_3x3.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05_box_3x3,Mean_s,img_size)

SaltAndPepper_Image_0_05_box_5x5 = cv2.imread("SaltAndPepper_Image_0_05_box_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05_box_5x5.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05_box_5x5,Mean_s,img_size)

SaltAndPepper_Image_0_05_median_3x3 = cv2.imread("SaltAndPepper_Image_0_05_median_3x3.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05_median_3x3.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05_median_3x3,Mean_s,img_size)

SaltAndPepper_Image_0_05_median_5x5 = cv2.imread("SaltAndPepper_Image_0_05_median_5x5.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05_median_5x5.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05_median_5x5,Mean_s,img_size)

SaltAndPepper_Image_0_05_close_to_open = cv2.imread("SaltAndPepper_Image_0_05_close_to_open.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05_close_to_open.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05_close_to_open,Mean_s,img_size)

SaltAndPepper_Image_0_05_open_to_close = cv2.imread("SaltAndPepper_Image_0_05_open_to_close.bmp", cv2.IMREAD_GRAYSCALE)
print("SaltAndPepper_Image_0_05_open_to_close.bmp")
Get_SNR(img,SaltAndPepper_Image_0_05_open_to_close,Mean_s,img_size)

