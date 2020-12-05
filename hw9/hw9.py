# %%
import numpy as np
import cv2
from tqdm import tqdm
import math

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
class Kernel:
    def __init__(self, init_list, origin):
        self.pattern = init_list
        self.origin = origin

    def get_directions(self):
        tmp_list = []
        for i in range(len(self.pattern)):
            for j in range(len(self.pattern[0])):
                direction = (i - self.origin[0], j - self.origin[1])
                tmp_list.append(direction)
        return tmp_list


def Roberts_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            r1_x = src[i][j]
            r1, r2, gradient = 0, 0, 0
            if i+1 < len(src):
                r2_x_1 = src[i+1][j]
            else:
                r2_x_1 = src[i][j]
            if j+1 < len(src[0]):
                r2_x = src[i][j+1]
            else:
                r2_x = src[i][j]
            if i+1 < len(src) and j+1 < len(src[0]):
                r1_x_1 = src[i+1][j+1]
            else:
                r1_x_1 = src[i][j]
            r1 = float(r1_x_1) - float(r1_x)
            r2 = float(r2_x_1) - float(r2_x)
            gradient = math.sqrt(r1*r1+r2*r2)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
def Prewitt_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            p1_x = src[i][j]

            if i+1 < len(src):
                bottom = src[i+1][j]
            else:
                bottom = src[i][j]

            if i-1 >= 0:
                top = src[i-1][j]
            else:
                top = src[i][j]

            if j+1 < len(src[0]):
                right = src[i][j+1]
            else:
                right = src[i][j]

            if j-1 >= 0:
                left = src[i][j-1]
            else:
                left = src[i][j]

            if i+1 < len(src) and j+1 < len(src[0]):
                bottom_right = src[i+1][j+1]
            elif j+1 < len(src[0]):
                bottom_right = src[i][j+1]
            elif i+1 < len(src):
                bottom_right = src[i+1][j]
            else:
                bottom_right = src[i][j]

            if i+1 < len(src) and j-1 >= 0:
                bottom_left = src[i+1][j-1]
            elif j-1 < len(src[0]):
                bottom_left = src[i][j-1] 
            elif i+1 < len(src):
                bottom_left = src[i+1][j] 
            else:
                bottom_left = src[i][j]

            if i-1 >= 0 and j+1 < len(src[0]):
                top_right = src[i-1][j+1]
            elif j+1 < len(src[0]):
                top_right = src[i][j+1] 
            elif i-1 >= 0:
                top_right = src[i-1][j] 
            else:
                top_right = src[i][j]


            if i-1 >= 0 and j-1 >= 0:
                top_left = src[i-1][j-1]
            elif j-1 >= 0:
                top_left = src[i][j-1] 
            elif i-1 >= 0:
                top_left = src[i-1][j] 
            else:
                top_left = src[i][j]
            p1 = float(bottom_left) + float(bottom) + float(bottom_right) - float(top_left) - float(top) - float(top_right)
            p2 = float(top_right) + float(right) + float(bottom_right) - float(top_left) - float(left) - float(bottom_left)
            gradient = math.sqrt(p1*p1+p2*p2)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
def Sobel_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            if i+1 < len(src):
                bottom = src[i+1][j]
            else:
                bottom = src[i][j]

            if i-1 >= 0:
                top = src[i-1][j]
            else:
                top = src[i][j]

            if j+1 < len(src[0]):
                right = src[i][j+1]
            else:
                right = src[i][j]

            if j-1 >= 0:
                left = src[i][j-1]
            else:
                left = src[i][j]

            if i+1 < len(src) and j+1 < len(src[0]):
                bottom_right = src[i+1][j+1]
            elif j+1 < len(src[0]):
                bottom_right = src[i][j+1]
            elif i+1 < len(src):
                bottom_right = src[i+1][j]
            else:
                bottom_right = src[i][j]

            if i+1 < len(src) and j-1 >= 0:
                bottom_left = src[i+1][j-1]
            elif j-1 < len(src[0]):
                bottom_left = src[i][j-1] 
            elif i+1 < len(src):
                bottom_left = src[i+1][j] 
            else:
                bottom_left = src[i][j]

            if i-1 >= 0 and j+1 < len(src[0]):
                top_right = src[i-1][j+1]
            elif j+1 < len(src[0]):
                top_right = src[i][j+1] 
            elif i-1 >= 0:
                top_right = src[i-1][j] 
            else:
                top_right = src[i][j]


            if i-1 >= 0 and j-1 >= 0:
                top_left = src[i-1][j-1]
            elif j-1 >= 0:
                top_left = src[i][j-1] 
            elif i-1 >= 0:
                top_left = src[i-1][j] 
            else:
                top_left = src[i][j]
            s1 = float(bottom_left) + 2*float(bottom) + float(bottom_right) - float(top_left) - 2*float(top) - float(top_right)
            s2 = float(top_right) + 2*float(right) + float(bottom_right) - float(top_left) - 2*float(left) - float(bottom_left)
            gradient = math.sqrt(s1*s1+s2*s2)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
def F_C_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            if i+1 < len(src):
                bottom = src[i+1][j]
            else:
                bottom = src[i][j]

            if i-1 >= 0:
                top = src[i-1][j]
            else:
                top = src[i][j]

            if j+1 < len(src[0]):
                right = src[i][j+1]
            else:
                right = src[i][j]

            if j-1 >= 0:
                left = src[i][j-1]
            else:
                left = src[i][j]

            if i+1 < len(src) and j+1 < len(src[0]):
                bottom_right = src[i+1][j+1]
            elif j+1 < len(src[0]):
                bottom_right = src[i][j+1]
            elif i+1 < len(src):
                bottom_right = src[i+1][j]
            else:
                bottom_right = src[i][j]

            if i+1 < len(src) and j-1 >= 0:
                bottom_left = src[i+1][j-1]
            elif j-1 < len(src[0]):
                bottom_left = src[i][j-1] 
            elif i+1 < len(src):
                bottom_left = src[i+1][j] 
            else:
                bottom_left = src[i][j]

            if i-1 >= 0 and j+1 < len(src[0]):
                top_right = src[i-1][j+1]
            elif j+1 < len(src[0]):
                top_right = src[i][j+1] 
            elif i-1 >= 0:
                top_right = src[i-1][j] 
            else:
                top_right = src[i][j]


            if i-1 >= 0 and j-1 >= 0:
                top_left = src[i-1][j-1]
            elif j-1 >= 0:
                top_left = src[i][j-1] 
            elif i-1 >= 0:
                top_left = src[i-1][j] 
            else:
                top_left = src[i][j]
            f1 = float(bottom_left) + math.sqrt(2*float(bottom)) + float(bottom_right) - float(top_left) - math.sqrt(2*float(top)) - float(top_right)
            f2 = float(top_right) + math.sqrt(2*float(right)) + float(bottom_right) - float(top_left) - math.sqrt(2*float(left)) - float(bottom_left)
            gradient = math.sqrt(f1*f1+f2*f2)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
def Kirsch_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            if i+1 < len(src):
                bottom = src[i+1][j]
            else:
                bottom = src[i][j]

            if i-1 >= 0:
                top = src[i-1][j]
            else:
                top = src[i][j]

            if j+1 < len(src[0]):
                right = src[i][j+1]
            else:
                right = src[i][j]

            if j-1 >= 0:
                left = src[i][j-1]
            else:
                left = src[i][j]

            if i+1 < len(src) and j+1 < len(src[0]):
                bottom_right = src[i+1][j+1]
            elif j+1 < len(src[0]):
                bottom_right = src[i][j+1]
            elif i+1 < len(src):
                bottom_right = src[i+1][j]
            else:
                bottom_right = src[i][j]

            if i+1 < len(src) and j-1 >= 0:
                bottom_left = src[i+1][j-1]
            elif j-1 < len(src[0]):
                bottom_left = src[i][j-1] 
            elif i+1 < len(src):
                bottom_left = src[i+1][j] 
            else:
                bottom_left = src[i][j]

            if i-1 >= 0 and j+1 < len(src[0]):
                top_right = src[i-1][j+1]
            elif j+1 < len(src[0]):
                top_right = src[i][j+1] 
            elif i-1 >= 0:
                top_right = src[i-1][j] 
            else:
                top_right = src[i][j]


            if i-1 >= 0 and j-1 >= 0:
                top_left = src[i-1][j-1]
            elif j-1 >= 0:
                top_left = src[i][j-1] 
            elif i-1 >= 0:
                top_left = src[i-1][j] 
            else:
                top_left = src[i][j]
            k0 = 5 * (float(top_right) + float(right) + float(bottom_right)) - 3 * (float(top) + float(top_left) + float(left) + float(bottom_left) + float(bottom))
            k1 = 5 * (float(top) + float(top_right) + float(right)) - 3 * (float(top_left) + float(left) + float(bottom_left) + float(bottom) + float(bottom_right))
            k2 = 5 * (float(top_left) + float(top) + float(top_right)) - 3 * (float(left) + float(bottom_left) + float(bottom) + float(bottom_right) + float(right))
            k3 = 5 * (float(left) + float(top_left) + float(top)) - 3 * (float(bottom_left) + float(bottom) + float(bottom_right) + float(right) + float(top_right))
            k4 = 5 * (float(bottom_left) + float(left) + float(top_left)) - 3 * (float(bottom) + float(bottom_right) + float(right) + float(top_right) + float(top))
            k5 = 5 * (float(bottom) + float(bottom_left) + float(left)) - 3 * (float(bottom_right) + float(right) + float(top_right) + float(top) + float(top_left))
            k6 = 5 * (float(bottom_right) + float(bottom) + float(bottom_left)) - 3 * (float(right) + float(top_right) + float(top) + float(top_left) + float(left))
            k7 = 5 * (float(right) + float(bottom_right) + float(bottom)) - 3 * (float(top_right) + float(top) + float(top_left) + float(left) + float(bottom_left))
            gradient = max(k0,k1,k2,k3,k4,k5,k6,k7)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
def Robinson_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            if i+1 < len(src):
                bottom = src[i+1][j]
            else:
                bottom = src[i][j]

            if i-1 >= 0:
                top = src[i-1][j]
            else:
                top = src[i][j]

            if j+1 < len(src[0]):
                right = src[i][j+1]
            else:
                right = src[i][j]

            if j-1 >= 0:
                left = src[i][j-1]
            else:
                left = src[i][j]

            if i+1 < len(src) and j+1 < len(src[0]):
                bottom_right = src[i+1][j+1]
            elif j+1 < len(src[0]):
                bottom_right = src[i][j+1]
            elif i+1 < len(src):
                bottom_right = src[i+1][j]
            else:
                bottom_right = src[i][j]

            if i+1 < len(src) and j-1 >= 0:
                bottom_left = src[i+1][j-1]
            elif j-1 < len(src[0]):
                bottom_left = src[i][j-1] 
            elif i+1 < len(src):
                bottom_left = src[i+1][j] 
            else:
                bottom_left = src[i][j]

            if i-1 >= 0 and j+1 < len(src[0]):
                top_right = src[i-1][j+1]
            elif j+1 < len(src[0]):
                top_right = src[i][j+1] 
            elif i-1 >= 0:
                top_right = src[i-1][j] 
            else:
                top_right = src[i][j]


            if i-1 >= 0 and j-1 >= 0:
                top_left = src[i-1][j-1]
            elif j-1 >= 0:
                top_left = src[i][j-1] 
            elif i-1 >= 0:
                top_left = src[i-1][j] 
            else:
                top_left = src[i][j]
            r0 = float(top_right) + 2* float(right) + float(bottom_right) - float(top_left) - 2 * float(left) - float(bottom_left)
            r1 = float(top) + 2* float(top_right) + float(right) - float(left) - 2 * float(bottom_left) - float(bottom)
            r2 = float(top_left) + 2* float(top) + float(top_right) - float(bottom_left) - 2 * float(bottom) - float(bottom_right)
            r3 = float(left) + 2* float(top_left) + float(top) - float(bottom) - 2 * float(bottom_right) - float(right)
            r4 = float(bottom_left) + 2* float(left) + float(top_left) - float(bottom_right) - 2 * float(right) - float(top_right)
            r5 = float(bottom) + 2* float(bottom_left) + float(left) - float(right) - 2 * float(top_right) - float(top)
            r6 = float(bottom_right) + 2* float(bottom) + float(bottom_left) - float(top_right) - 2 * float(top) - float(top_left)
            r7 = float(right) + 2* float(bottom_right) + float(bottom) - float(top) - 2 * float(top_left) - float(left)
            gradient = max(r0,r1,r2,r3,r4,r5,r6,r7)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img

pattern = []
def Nevati_Babu_operator(src, threshold):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.uint8)
    
    for i in range(len(src)):
        for j in range(len(src[0])):
            mask = np.zeros((5, 5), dtype=np.uint8)
            mask[2][2] = src[i][j]

            if i+1 < len(src):
                mask[3][2] = src[i+1][j]
            else:
                mask[3][2] = src[i][j]

            if i-1 >= 0:
                mask[1][2] = src[i-1][j]
            else:
                mask[1][2] = src[i][j]

            if j+1 < len(src[0]):
                mask[2][3] = src[i][j+1]
            else:
                mask[2][3] = src[i][j]

            if j-1 >= 0:
                mask[2][1] = src[i][j-1]
            else:
                mask[2][1] = src[i][j]

            if i+1 < len(src) and j+1 < len(src[0]):
                mask[3][3] = src[i+1][j+1]
            elif j+1 < len(src[0]):
                mask[3][3] = src[i][j+1]
            elif i+1 < len(src):
                mask[3][3] = src[i+1][j]
            else:
                mask[3][3] = src[i][j]

            if i+1 < len(src) and j-1 >= 0:
                mask[3][1] = src[i+1][j-1]
            elif j-1 < len(src[0]):
                mask[3][1] = src[i][j-1] 
            elif i+1 < len(src):
                mask[3][1] = src[i+1][j] 
            else:
                mask[3][1] = src[i][j]

            if i-1 >= 0 and j+1 < len(src[0]):
                mask[1][3] = src[i-1][j+1]
            elif j+1 < len(src[0]):
                mask[1][3] = src[i][j+1] 
            elif i-1 >= 0:
                mask[1][3] = src[i-1][j] 
            else:
                mask[1][3] = src[i][j]


            if i-1 >= 0 and j-1 >= 0:
                mask[1][1] = src[i-1][j-1]
            elif j-1 >= 0:
                mask[1][1] = src[i][j-1] 
            elif i-1 >= 0:
                mask[1][1] = src[i-1][j] 
            else:
                mask[1][1] = src[i][j]

            if i-2 >= 0:
                if j-1 >= 0:
                    mask[0][1] = src[i-2][j-1]
                else:
                    mask[0][1] = mask[1][1]
                mask[0][2] = src[i-2][j]
                if j+1 < len(src):
                    mask[0][3] = src[i-2][j+1]
                else:
                    mask[0][3] = mask[1][3]
            else:
                mask[0][1] = mask[1][1]
                mask[0][2] = mask[1][2]
                mask[0][3] = mask[1][3]

            if i+2 < len(src):
                if j-1 >= 0:
                    mask[4][1] = src[i+2][j-1]
                else:
                    mask[4][1] = mask[3][1]
                mask[4][2] = src[i+2][j]
                if j+1 < len(src):
                    mask[4][3] = src[i+2][j+1]
                else:
                    mask[4][3] = mask[3][3]
            else:
                mask[4][1] = mask[3][1]
                mask[4][2] = mask[3][2]
                mask[4][3] = mask[3][3]
            if j-2 >= 0:
                if i-1 >= 0:
                    mask[1][0] = src[i-1][j-2]
                else:
                    mask[1][0] = mask[1][1]
                mask[2][0] = src[i][j-2]
                if i+1 < len(src):
                    mask[3][0] = src[i+1][j-2]
                else:
                    mask[3][0] = mask[3][1]
            else:
                mask[1][0] = mask[1][1]
                mask[2][0] = mask[2][1]
                mask[3][0] = mask[3][1]
            if j+2 < len(src):
                if i-1 >= 0:
                    mask[1][4] = src[i-1][j+2]
                else:
                    mask[1][4] = mask[1][3]
                mask[2][4] = src[i][j+2]
                if i+1 < len(src):
                    mask[3][4] = src[i+1][j+2]
                else:
                    mask[3][4] = mask[3][3]
            else:
                mask[1][4] = mask[1][3]
                mask[2][4] = mask[2][3]
                mask[3][4] = mask[3][3]
            
            if i-2 >= 0 and j-2 >= 0 :
                mask[0][0] = src[i-2][j-2]
            else:
                mask[0][0] = mask[1][1]

            if i-2 >= 0 and j+2 < len(src):
                mask[0][4] = src[i-2][j+2]
            else:
                mask[0][4] = mask[1][3]

            if i+2 < len(src) and j-2 >= 0:
                mask[4][0] = src[i+2][j-2]
            else:
                mask[4][0] = mask[3][1]
            
            if i+2 < len(src)and j+2 < len(src):
                mask[4][4] = src[i+2][j+2]
            else:
                mask[4][4] = mask[3][3]
            
            pattern0 = [
                [100,100,100,100,100],
                [100,100,100,100,100],
                [  0,  0,  0,  0,  0],
                [-100,-100,-100,-100,-100],
                [-100,-100,-100,-100,-100]
                ]
            pattern30 = [
                [100,100,100,100,100],
                [100,100,100,78,-32],
                [100, 92,  0,-92,-100],
                [ 37, -78,-100,-100,-100],
                [-100,-100,-100,-100,-100]
                ]
            pattern60 = [
                [100,100,100,32,-100],
                [100,100,92,-78,-100],
                [100,100,0,-100,-100],
                [-100,78,-92,-100,-100],
                [100,-32,-100,-100,-100]
                ]
            pattern_90 = [
                [-100,-100,0,100,100],
                [-100,-100,0,100,100],
                [-100,-100,0,100,100],
                [-100,-100,0,100,100],
                [-100,-100,0,100,100]
                ]
            pattern_60 = [
                [-100,32,100,100,100],
                [-100,-78,92,100,100],
                [-100,-100,0,100,100],
                [-100,-100,-92,78,100],
                [-100,-100,-100,-32,100]
                ]
            pattern_30 = [
                [100,100,100,100,100],
                [-32,78,100,100,100],
                [-100,-92,0,92,100],
                [-100,-100,-100,-78,32],
                [-100,-100,-100,-100,-100]
                ]
            N0 = sum([int(mask[i][j] * pattern0[i][j]) for i in range(len(mask)) for j in range(len(mask[0]))])
            N1 = sum([int(mask[i][j] * pattern30[i][j]) for i in range(len(mask)) for j in range(len(mask[0]))])
            N2 = sum([int(mask[i][j] * pattern60[i][j]) for i in range(len(mask)) for j in range(len(mask[0]))])
            N3 = sum([int(mask[i][j] * pattern_90[i][j]) for i in range(len(mask)) for j in range(len(mask[0]))])
            N4 = sum([int(mask[i][j] * pattern_60[i][j]) for i in range(len(mask)) for j in range(len(mask[0]))])
            N5 = sum([int(mask[i][j] * pattern_30[i][j]) for i in range(len(mask)) for j in range(len(mask[0]))])
            gradient = max(N0,N1,N2,N3,N4,N5)
            if gradient >= threshold:   
                new_img[i][j] = 0
            else:
                new_img[i][j] = 255

    return new_img
Robert_12 = Roberts_operator(img, 12)
Robert_30 = Roberts_operator(img, 30)
Perwitt = Prewitt_operator(img,24)
Sobel = Sobel_operator(img,38)
F_C = F_C_operator(img,30)
Kirsch = Kirsch_operator(img,135)
Robinson = Robinson_operator(img,43)
Nevati_Babu = Nevati_Babu_operator(img,12500)


cv2.imwrite('Robert_12.bmp', Robert)
cv2.imwrite('Robert_30.bmp', Robert_30)
cv2.imwrite('Perwitt.bmp', Perwitt)
cv2.imwrite('Sobel.bmp', Sobel)
cv2.imwrite('F_C.bmp', F_C)
cv2.imwrite('Kirsch.bmp', Kirsch)
cv2.imwrite('Robinson.bmp', Robinson)
cv2.imwrite('Nevati_Babu.bmp', Nevati_Babu)


# cv2.imshow("test", Robert)


cv2.waitKey(0)
# %%
