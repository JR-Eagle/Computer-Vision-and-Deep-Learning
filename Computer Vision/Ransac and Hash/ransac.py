"""
@author: Rai
Random Sample Consensus (RANSAC)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from Least_Squares import least_squares


nums = 300  # Number of samples
X = np.linspace(0, 50, nums)
A, B = 2, 10
Y = A * X + B

# Random noise in the line
random_x = X + np.random.normal(size=nums)
random_y = Y + np.random.normal(size=nums)

# Arbitrary noise
x_noise = []
y_noise = []
for i in range(nums):
    x_noise.append(random.uniform(0, 50))
    y_noise.append(random.uniform(10, 110))
x_noise = np.array(x_noise)
y_noise = np.array(y_noise)
random_X = np.hstack([random_x, x_noise])
random_Y = np.hstack([random_y, y_noise])

# Plotting the scatter graph
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('RANSAC')
ax.scatter(random_x, random_y, c='k', label='RANSAC_data')
ax.scatter(x_noise, y_noise, c='y', label='noise data')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Estimating the model using the RANSAC algorithm
iteration = 10000  # Maximum number of iterations
diff = 0.01  # Error between the model and real data
best_a = 0  # Best model's slope
best_b = 0  # Best model's intercept
max_inlier = 0  # Maximum number of inliers
p = 0.99  # Desired probability of getting the correct model
for i in range(iteration):
    # Randomly selecting two points from the data to solve the model
    sample_id = random.sample(range(nums*2), 2)
    i, j = sample_id
    x1 = random_X[i]
    x2 = random_X[j]
    y1 = random_Y[i]
    y2 = random_Y[j]

    # From y = a*x + b -> a, b
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    # Counting the number of inliers
    total_inlier = 0
    for i in range(nums * 2):
        y_pred = a * random_X[i] + b
        error = abs(y_pred - random_Y[i])
        if error < diff:
            total_inlier += 1

    # Checking if the current model is better than the previous one by inlier count
    if total_inlier > max_inlier:
        iteration = math.log(1 - p) / math.log(1 - pow((total_inlier / nums * 2), 2))
        max_inlier = total_inlier
        best_a = a
        best_b = b

    # Checking if inlier count exceeds 95% of data
    if total_inlier > int(nums * 0.95):
        break

Y_pred = best_a * random_X + best_b
Y = A * random_X + B
k, b = least_squares(random_X, random_Y)
Y_linear = k * random_X + b
print('best_a:', best_a)
print('best_b:', best_b)
ax.plot(random_X, Y_pred, color='r', label='RANSAC fit')
ax.plot(random_X, Y, color='b', label='Real data')
ax.plot(random_X, Y_linear, color='c', label='Linear fit')
ax.legend(loc='upper left')
plt.savefig('RANSAC.png')
plt.show()
