# -*- coding: utf-8 -*-
"""
@author:rai
Gaussian Noise
"""

import numpy as np
import cv2
import random

def add_gaussian_noise(image, mean, sigma, percentage):
    """
    Introduces Gaussian noise to an image.
    
    Parameters:
    - image: Input image.
    - mean: Mean value of the Gaussian noise.
    - sigma: Standard deviation of the Gaussian noise.
    - percentage: Ratio of pixels that will be affected by the noise.
    
    Returns:
    - noised_image: Image with Gaussian noise added.
    """
    
    noised_image = image.copy()
    
    # Calculate the number of pixels to be noised
    total_pixels_to_noise = int(percentage * image.shape[0] * image.shape[1])
    
    for _ in range(total_pixels_to_noise):
        # Randomly select a pixel
        rand_x = random.randint(0, image.shape[0] - 1)
        rand_y = random.randint(0, image.shape[1] - 1)
        
        # Add Gaussian noise to the selected pixel
        noised_image[rand_x, rand_y] += random.gauss(mean, sigma)
        
        # Clip the pixel value to be between 0 and 255
        noised_image[rand_x, rand_y] = np.clip(noised_image[rand_x, rand_y], 0, 255)

    return noised_image

if __name__ == '__main__':
    color = False
    if color:
        img = cv2.imread('lenna.png')
        # Add Gaussian noise to the color image
        noised_image = add_gaussian_noise(img, 2, 4, 0.8)
        img_merge = np.hstack([img, noised_image])
        cv2.imwrite('Gaussian Image_Color.png', img_merge)
    else:
        img = cv2.imread('lenna.png', 0)
        # Add Gaussian noise to the grayscale image
        noised_image = add_gaussian_noise(img, 2, 4, 0.8)
        img_merge = np.hstack([img, noised_image])
        cv2.imwrite('Gaussian Image_Gray.png', img_merge)
    cv2.imshow('Left: Original Image, Right: Gaussian Image', img_merge)
    cv2.waitKey(0)
