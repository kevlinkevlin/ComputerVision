# %%
import numpy as np
import cv2
from tqdm import tqdm
import operator
import csv

img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

new_height = len(img[0])//8
new_width = len(img)//8

small_img = np.zeros((new_height, new_width), dtype=np.uint8)
ans_img = np.zeros((new_height, new_width), dtype=np.str)


def h(b, c, d, e):
    q, r, s = 0, 0, 0
    if b != c:
        s = 1
    elif d == b and e == b:
        r = 1
    else:
        q = 1
    return (q, r, s)


def Yokoi(src, i, j):
    b, c, d, e = 0, 0, 0, 0
    if i-1 >= 0 and j+1 < len(src[0]):
        b, c, d, e = src[i][j], src[i][j+1], src[i-1][j+1], src[i-1][j]
    if i-1 >= 0:
         e =  src[i-1][j]
    if j+1 < len(src[0]):
        c = src[i][j+1]
    x1 = h(b, c, d, e)

    b, c, d, e = 0, 0, 0, 0
    if i-1 >= 0 and j-1 >= 0:
        b, c, d, e = src[i][j], src[i-1][j], src[i-1][j-1], src[i][j-1]
    if i-1 >= 0:
        c = src[i-1][j]
    if j-1 >= 0:
         e =  src[i][j-1]

    x2 = h(b, c, d, e)

    b, c, d, e = 0, 0, 0, 0
    if j-1 >= 0 and i+1 < len(src):
        b, c, d, e = src[i][j], src[i][j-1], src[i+1][j-1], src[i+1][j]
    if i+1 < len(src):
        e =  src[i+1][j]
    if j-1 >= 0:
        c= src[i][j-1]
    x3 = h(b, c, d, e)

    b, c, d, e = 0, 0, 0, 0
    if i+1 < len(src) and j+1 < len(src[0]):
        b, c, d, e = src[i][j], src[i+1][j], src[i+1][j+1], src[i][j+1]

    if i+1 < len(src):
        c = src[i+1][j]
    if j+1 < len(src[0]):
        e =  src[i][j+1]
    x4 = h(b, c, d, e)


    result = tuple(map(lambda i, j, k, l: i + j + k + l, x1, x2, x3, x4))
    r,q = result[1],result[0]
    if r == 4:
        return '5'
    elif q > 0:
        return str(q)
    return ' '


for i in range(len(small_img)):
    for j in range(len(small_img[0])):
        offset_x, offset_y = 8*i, 8*j
        if img[offset_x][offset_y] < 128:
            small_img[i][j] = 0
        else:
            small_img[i][j] = 255
for i in range(len(ans_img)):
    for j in range(len(ans_img[0])):
        if small_img[i][j] == 0:
            ans_img[i][j] = ' '
            continue
        ans_img[i][j] = Yokoi(small_img, i, j)

# cv2.imshow("test", ans_img)
np.savetxt('ans.csv',ans_img, delimiter=' ',fmt = '%s')

cv2.waitKey(0)
# %%
