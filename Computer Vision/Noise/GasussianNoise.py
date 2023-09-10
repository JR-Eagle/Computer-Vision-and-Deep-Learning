# -*- coding: utf-8 -*-
"""
@author:rai
Gaussian Noise
"""

import cv2
import numpy as np
import random


def add_gaussian_noise(image, mean, sigma, ratio):
    """
    Introduces Gaussian noise to an image.

    Parameters:
    - image: Input image.
    - mean: Mean value of the Gaussian noise.
    - sigma: Standard deviation of the Gaussian noise.
    - ratio: Ratio of pixels that will be affected by the noise.

    Returns:
    - noised_image: Image with Gaussian noise added.
    """

    # Make a copy of the original image
    noised_image = image.copy()

    # Calculate the total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the number of pixels to be noised
    pixels_to_noise = int(np.floor(total_pixels * ratio))

    # Generate unique random numbers to determine which pixels will get noise
    random_pixels = np.array(random.sample(range(0, total_pixels), pixels_to_noise))
    x_coords = random_pixels // image.shape[0]
    y_coords = random_pixels % image.shape[1]

    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, sigma, image.shape)

    # Apply the Gaussian noise to the randomly selected pixels
    noised_image[x_coords, y_coords] = (noised_image + gaussian_noise)[x_coords, y_coords]

    # Clip the image values to be between 0 and 255 and convert to uint8 format
    noised_image = np.uint8(np.clip(noised_image, 0, 255))

    return noised_image


if __name__ == '__main__':
    # Flag to decide if the input image is in color or grayscale
    is_color = True

    # Read the input image based on the color flag
    if is_color:
        input_image = cv2.imread('lenna.png')
        noised_image = add_gaussian_noise(input_image, 1, 4, 0.9)
        merged_image = np.hstack([input_image, noised_image])
        cv2.imwrite('Gaussian_Image_Color.png', merged_image)
    else:
        input_image = cv2.imread('lenna.png', 0)
        noised_image = add_gaussian_noise(input_image, 1, 4, 0.01)
        merged_image = np.hstack([input_image, noised_image])
        cv2.imwrite('Gaussian_Image_Gray.png', merged_image)

    # Display the merged image: Original on the left and Noised on the right
    cv2.imshow('Left: Original Image, Right: Noised Image', merged_image)
    cv2.waitKey(0)
