"""
@author: Rai
Manual implementation of Canny edge detection.
"""

import cv2
import numpy as np
import math

def canny_edge_detection(sigma):
    """Manually implements Canny edge detection on an image."""
    
    # Load the image
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
    gauss_kernel = np.array([[math.exp(c2 * (vx ** 2 + vy ** 2)) for vx in v] for vy in v])
    gauss_filter = gauss_kernel / gauss_kernel.sum()
    img_padded = np.pad(img_gray, ksize // 2)
    dx, dy = img.shape[:2]
    img_gauss = np.array([[np.sum(img_padded[i:i + ksize, j:j + ksize] * gauss_filter) for j in range(dy)] for i in range(dx)])
    img_gauss = np.clip(img_gauss, 0, 255)

    # 2. Calculate gradient using Sobel operator
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # For vertical edges
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # For horizontal edges

    # Continue with the main gradient calculation...
    
    # 3. Non-maximum suppression
    img_NMS = np.zeros(img_gauss.shape)

    # 4. Dual threshold detection and edge linking
    edge_lower = img_gauss.mean() * 0.5
    edge_upper = edge_lower * 3
    img_cany = img_NMS.copy()
    img_cany[img_cany >= edge_upper] = 255  # Weak Edge
    img_cany[img_cany <= edge_lower] = 0   # Supression
    near_field_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            # Weak Edge
            if img_cany[i, j] > edge_lower and img_cany[i,j] < edge_upper:
                tmp = img_cany[i-1:i+2, j-1:j+2] * near_field_8
                if np.max(tmp) >= edge_upper:
                    img_cany[i, j] = 255
                else:
                    img_cany[i, j] = 0
    
    
if __name__ == "__main__":
    canny_edge_detection(sigma=0.5)
