# -*- coding: utf-8 -*-
"""
@author:rai
Gaussian Noise
"""

import numpy as np
import cv2
import random

def GaussianNoise(img, mean, sigma, ratio):
    """
    This function introduces Gaussian noise to an image.
    
    Parameters:
    - img: input image
    - mean: mean value of the Gaussian noise
    - sigma: standard deviation of the Gaussian noise
    - ratio: ratio of pixels that will be affected by the noise
    
    Returns:
    - noise_img: image with Gaussian noise added
    """
    
    noise_img = img.copy()
    pixels_total = img.shape[0] * img.shape[1]
    noise_num = int(np.floor(pixels_total * ratio))
    rand_num = np.array(random.sample(range(0, pixels_total), noise_num))  # Generate non-repeating random numbers
    x = rand_num // img.shape[1]
    y = rand_num % img.shape[1]
    
    gauss_img = np.random.normal(mean, sigma, img.shape)
    noise_img[x, y] = (noise_img + gauss_img)[x, y]
    noise_img = np.uint8(np.clip(noise_img, 0, 255))

    return noise_img


if __name__ == '__main__':
    color = False
    if color:
        img = cv2.imread('lenna.png')
        img_g
