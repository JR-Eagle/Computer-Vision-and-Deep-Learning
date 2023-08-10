# -*- coding utf-8 -*-
'''
@author:rai
PCA (Principal Component Analysis)
'''

import numpy as np

class MyPCA(object):
    def __init__(self, X, dim):
        self.X = X
        self.dim = dim
        self.centraX = self.centralization()  # Data centralization
        self.C = self.convariationCalc()  # Compute covariance
        self.R = self.eigenvalueCalc()  # Compute eigenvalues and eigenvectors
        self.X_new = self.transform()  # Compute data after dimensionality reduction

    def centralization(self):
        X_mean = [np.mean(v) for v in self.X.T]  # Compute mean of each feature of sample X
        print('Sample data feature mean values:\n', X_mean)
        centraX = self.X - X_mean
        print('Centralized sample data:\n', centraX)
        return centraX

    def convariationCalc(self):
        C = np.dot(self.centraX.T, self.centraX) / (np.shape(self.centraX)[0] - 1)
        print("Sample data covariance matrix:\n", C)
        return C

    def eigenvalueCalc(self):
        eig_value, eig_vector = np.linalg.eig(self.C)
        print("Eigenvalues of the sample data covariance matrix:\n", eig_value)
        print("Eigenvectors of the sample data covariance matrix:\n", eig_vector)
        id_des = np.argsort(eig_value)[::-1]
        eig_vector_tranf = np.array([eig_vector[:,i] for i in id_des[:self.dim]]).T
        print("%d-order dimension reduction transformation matrix:\n" % self.dim, eig_vector_tranf)
        return eig_vector_tranf

    def transform(self):
        X_new = np.dot(self.X, self.R)
        print("Data after dimensionality reduction:\n", X_new)
        return X_new

if __name__ == "__main__":
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])

    dim = np.shape(X)[1] - 1
    MyPCA(X, dim)
