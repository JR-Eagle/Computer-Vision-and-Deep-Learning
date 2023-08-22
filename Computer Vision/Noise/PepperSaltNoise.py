# -*- coding: utf-8 *-*
"""
@author: Rai
Salt and Pepper Noise
"""

import random
import numpy as np
import cv2

def PepperSaltNoise(img, ratio):
    """
    This function introduces salt and pepper noise to an image.
    
    Parameters:
    - img: input image
    - ratio: ratio of pixels that will be affected by the noise
    
    Returns:
    - noise_img: image with salt and pepper noise added
    """
    
    noise_img = img.copy()
    pixel_total = img.shape[0] * img.shape[1]
    noise_num = int(pixel_total * ratio)
    rand_point = random.sample(range(0, pixel_total), noise_num)  # Generate non-repeating random numbers
    for point in rand_point:
        x = point // img.shape[0]
        y = point % img.shape[1]
        noise = random.random()
        noise_img[x, y] = 0 if noise < 0.5 else 255

    return noise_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_salt = PepperSaltNoise(img, 0.05)
    color = True
    if color:
        img_merge = np.hstack([img, img_salt])
        cv2.imwrite('Salt and Pepper Image_Color.png', img_merge)
    else:
        img_merge = np.hstack([img[:, :, 0], img_salt[:, :, 0]])
        cv2.imwrite('Salt and Pepper Image_Gray.png', img_merge)
    cv2.imshow('Left: Original Image, Right: Salt and Pepper Image', img_merge)
    cv2.waitKey(0)
