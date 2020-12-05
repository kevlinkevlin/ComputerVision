# %%
import numpy as np
import cv2
from tqdm import tqdm


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


for i in range(len(img)):
    for j in range(len(img[0])):
        if img[i][j] < 128:
            img[i][j] = 0
        else:
            img[i][j] = 255


# (a) Dilation
def dilation(src, kernel, directions):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            old = src[i][j]
            if old == 0:
                continue
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if new_i >= 0 and new_i < len(src) and new_j >= 0 and new_j < len(src[0]):
                    new_img[new_i][new_j] = 255

    return new_img

# (b) Erosion


def erosion(src, kernel, directions):
    new_img = np.zeros((len(src), len(src[0])), dtype=np.uint8)
    for i in range(len(src)):
        for j in range(len(src[0])):
            check = True
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if new_i >= 0 and new_i < len(src) and new_j >= 0 and new_j < len(src[0]):
                    if src[new_i][new_j] == 255:
                        continue
                    else:
                        check = False
                    break

            if check:
                new_img[i][j] = 255

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


octo_kernel = Kernel(octo_kernel_pattern, (2, 2))
octo_directions = octo_kernel.get_directions()
L_J_kernel = Kernel(L_shape_kernel_pattern, (0, 1))
L_J_kernel_directions = L_J_kernel.get_directions()
L_K_kernel = Kernel(L_shape_kernel_pattern, (1, 0))
L_K_kernel_directions = L_K_kernel.get_directions()

img_a = dilation(img, octo_kernel, octo_directions)
img_b = erosion(img, octo_kernel, octo_directions)
img_c = opening(img, octo_kernel, octo_directions)
img_d = closing(img, octo_kernel, octo_directions)
img_e = hit_and_miss(img, L_J_kernel, L_J_kernel_directions,
                     L_K_kernel, L_K_kernel_directions)

cv2.imwrite('img_a.bmp', img_a)
cv2.imwrite('img_b.bmp', img_b)
cv2.imwrite('img_c.bmp', img_c)
cv2.imwrite('img_d.bmp', img_d)
cv2.imwrite('img_e.bmp', img_e)
cv2.waitKey(0)
# %%
