"""
@author: Rai
Least Squares Method
"""

def least_squares(x, y):
    """
    x: numpy.array
    y: numpy.array
    """
    xy_sum = x*y.sum()
    x_sum = x.sum()
    y_sum = y.sum()
    xx_sum = x*x.sum()

    # Calculate slope and intercept
    n = len(x)
    k = (x_sum * y_sum - n * xy_sum) / (x_sum ** 2 - xx_sum * n)
    b = (y_sum - k * x_sum) / n
    return k, b
