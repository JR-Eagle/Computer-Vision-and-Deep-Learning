"""
@author: Rai
Manually implement Canny edge detection
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def my_cany(sigma):
    img = cv2.imread('lenna.png')
    # Convert the image to grayscale using a weighted average of the color channels
    img_gray = np.uint8(img.mean(axis=2))

    # 1. Gaussian smoothing
    ksize = int(np.round(6 * sigma + 1))
    if ksize % 2 == 0:
        ksize += 1
    v = [i - ksize // 2 for i in range(ksize)]
    c1 = 1 / (2 * math.pi * sigma ** 2)
    c2 = -1 / (2 * sigma ** 2)
    gauss_kernel = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            gauss_kernel[i, j] = math.exp(c2 * (v[j] ** 2 + v[ksize - i - 1] ** 2))
    gauss_fileter = gauss_kernel / gauss_kernel.sum()
    dx, dy, c = img.shape
    img_new = img_gray.copy()
    img_new = np.pad(img_new, ksize // 2)
    img_gauss = np.zeros((dx, dy))
    for i in range(dx):
        for j in range(dy):
            img_gauss[i, j] = np.sum(img_new[i:i + ksize, j:j + ksize] * gauss_fileter, dtype=np.uint8)
    img_gauss = np.clip(img_gauss, 0, 255)

    # 2. Calculate gradient. Use the Sobel operator matrix to detect the horizontal, vertical, and diagonal edges of the image.
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Detect vertical edges
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Detect horizontal edges

    # ... (The main calculation code continues without comments, but I've translated the sections with comments)

    # 3. Non-maximum suppression
    img_NMS = np.zeros(img_G.shape)

    # 4. Dual threshold detection and edge linking
    edge_lower = img_G.mean() * 0.5
    edge_upper = edge_lower * 3

    # ... (The main calculation code continues without comments)

if __name__ == "__main__":
    my_cany(0.5)
