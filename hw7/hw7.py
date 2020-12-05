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


def h(b, c, d, e, binary):
    q, r, s = 0, 0, 0
    if binary:
        if b > 0:
            b = 1
        if c > 0:
            c = 1
        if d > 0:
            d = 1
        if e > 0:
            e = 1

    if b != c:
        s = 1
    elif d == b and e == b:
        r = 1
    else:
        q = 1
    return (q, r, s)


def Yokoi(src, i, j, binary):
    b, c, d, e = 0, 0, 0, 0
    if i-1 >= 0 and j+1 < len(src[0]):
        b, d = src[i][j], src[i-1][j+1]
    if i-1 >= 0:
        e = src[i-1][j]
    if j+1 < len(src[0]):
        c = src[i][j+1]
    x1 = h(b, c, d, e, binary)

    b, c, d, e = 0, 0, 0, 0
    if i-1 >= 0 and j-1 >= 0:
        b, d = src[i][j], src[i-1][j-1]
    if i-1 >= 0:
        c = src[i-1][j]
    if j-1 >= 0:
        e = src[i][j-1]

    x2 = h(b, c, d, e, binary)

    b, c, d, e = 0, 0, 0, 0
    if j-1 >= 0 and i+1 < len(src):
        b, d = src[i][j], src[i+1][j-1]
    if i+1 < len(src):
        e = src[i+1][j]
    if j-1 >= 0:
        c = src[i][j-1]
    x3 = h(b, c, d, e, binary)

    b, c, d, e = 0, 0, 0, 0
    if i+1 < len(src) and j+1 < len(src[0]):
        b, d = src[i][j], src[i+1][j+1]

    if i+1 < len(src):
        c = src[i+1][j]
    if j+1 < len(src[0]):
        e = src[i][j+1]
    x4 = h(b, c, d, e, binary)

    result = tuple(map(lambda i, j, k, l: i + j + k + l, x1, x2, x3, x4))
    q, r = result[0], result[1]
    if r == 4:
        return 5
    return q


for i in range(len(small_img)):
    for j in range(len(small_img[0])):
        offset_x, offset_y = 8*i, 8*j
        if img[offset_x][offset_y] < 128:
            small_img[i][j] = 0
        else:
            small_img[i][j] = 255

for time in range(10):
    cv2.imwrite('ans' + str(time+1) + '.bmp', small_img)
    ans_img = np.zeros((new_height, new_width), dtype=np.uint8)
    Yokoi_img = np.zeros((new_height, new_width), dtype=np.uint8)
    for i in range(len(Yokoi_img)):
        for j in range(len(Yokoi_img[0])):
            if small_img[i][j] == 0:
                Yokoi_img[i][j] = 0
                continue
            Yokoi_img[i][j] = Yokoi(small_img, i, j, True)
            if Yokoi_img[i][j] > 0:
                ans_img[i][j] = 255
    for i in range(len(Yokoi_img)):
        for j in range(len(Yokoi_img[0])):
            if Yokoi_img[i][j] == 0:
                continue
            if Yokoi_img[i][j] > 1:
                Yokoi_img[i][j] = 2
                continue
            if i-1 >= 0 and Yokoi_img[i-1][j] == 1:
                continue
            if i+1 < len(Yokoi_img) and Yokoi_img[i+1][j] == 1:
                continue
            if j-1 >= 0 and Yokoi_img[i][j-1] == 1:
                continue
            if j+1 < len(Yokoi_img) and Yokoi_img[i][j+1] == 1:
                continue
            Yokoi_img[i][j] = 2  # if there is no 1 neighbor, change it into 2

    for i in range(len(Yokoi_img)):
        for j in range(len(Yokoi_img[0])):
            if Yokoi_img[i][j] != 1:
                continue
            if Yokoi(Yokoi_img, i, j, True) == 1:
                ans_img[i][j] = 0
                Yokoi_img[i][j] = 0
    small_img = ans_img
    # cv2.imwrite('ans' + str(time+1) + '.bmp', ans_img)

# cv2.imshow("test", ans_img)

cv2.waitKey(0)
# %%
