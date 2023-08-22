"""
@author: Rai
K-Means Clustering: Color Image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
retval, bestLabels, centers =cv2.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])

Input:
    data: Clustering data, preferably of type np.float32, an N-dimensional set of points with each feature in a column.
    K: Number of clustering groups.
    bestLabels: An output integer array used to store the cluster index for each sample.
    criteria: The mode choice for iteration termination, which is a tuple with three elements. Format: (type, max_iter, epsilon).
        Where, type has the following modes:
         - cv2.TERM_CRITERIA_EPS: Stops when the precision (error) reaches epsilon.
         - cv2.TERM_CRITERIA_MAX_ITER: Stops when the number of iterations exceeds max_iter.
         - cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER: Combined, stops when either condition is met.
    attempts: Number of times the kmeans algorithm is run, and the best result labels are returned by the algorithm.
    flags: Choice of initial centers. Two methods are cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS.
          - cv2.KMEANS_PP_CENTERS: Uses KMeans++ algorithm, where the basic principle during cluster center initialization is to have the initial cluster centers as far apart as possible.
          - cv2.KMEANS_RANDOM_CENTERS: Random selection.
    centers: Output matrix of cluster centers, with each cluster center being a row of data.

Output:
    compactness: Compactness, returns the sum of squared distances from each point to its assigned center.
    labels: Resultant labels, where each member is marked with a group number like 0, 1, 2, 3, 4...etc.
    centers: An array consisting of the centers of the clusters.
"""

img = cv2.imread('./lenna.png')
h, w, _ = img.shape

# Convert image data into a single row
data = img.reshape((h * w, 3))
data = np.float32(data)

# Number of clusters
# k_array = [2, 4, 8, 16, 32, 64, 128]
k_array = [2, 4, 8, 16, 32]

# Choice of iteration mode
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

# Choice for initial center
flags = cv2.KMEANS_PP_CENTERS

# Number of KMeans iterations
attempts = 10

compactness_a, labels, centers = [], [], []
for k in k_array:
    compactness, label, center = cv2.kmeans(data, k, None, criteria, attempts, flags)
    compactness_a.append(compactness)
    labels.append(label)
    centers.append(center)

imgs_dst = []
for i, (center, label) in enumerate(zip(centers, labels)):
    center = np.uint8(center)
    img_dst = center[label.flatten()]
    img_dst = img_dst.reshape((h, w, 3))
    imgs_dst.append(img_dst)

titles = ['Original Image', 'KMeans ']
images = [img]
images.extend(imgs_dst)

plt.figure(figsize=(40, 25))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    if i == 0:
        plt.title(titles[0], fontsize=60, color='blue')
    else:
        plt.title(titles[1] + 'k=' + str(k_array[i-1]), fontsize=60, color='blue')
    plt.axis('off')
plt.savefig('KMeans_RGB Image.png')
plt.show()
