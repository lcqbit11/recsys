#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

"""
矩阵分解.
Created by lichangqing03 on 2019/08/05.
"""

def load_data(file_name):
    file = open(file_name, 'rb')
    data = []
    for line in file:
        line = line.strip().split()
        arr = []
        for i in line:
            arr.append(float(i))
        data.append(arr)
    return data

def gra_descent(mn_data, K, alpha, beta):
    data_matrix = np.mat(mn_data)
    m, n = data_matrix.shape
    P = np.random.random([m, K])
    Q = np.random.random([K, n])

    epoch_num = 1000
    p_gra = 0
    q_gra = 0
    for epoch_i in range(epoch_num):
        for i in range(m):
            for j in range(n):
                error = data_matrix[i][j]
                for k in range(K):
                    error -= P[i][k]*Q[k][j]
                for k in range(K):
                    p_gra = -2*error*P[i][k] + beta*Q[k][j]
                    q_gra = -2*error*Q[k][j] + beta*P[i][k]
                    P[i][k] -= alpha*p_gra
                    Q[k][j] -= alpha*q_gra

        loss = 0
        for i in range(m):
            for j in range(n):
                error = 0
                for k in range(K):
                    error += P[i][k]*Q[k][j]
                loss = (data_matrix[i][j] - error)*(data_matrix[i][j] - error)
                for k in range(K):
                    loss += beta*(P[i][k]*P[i][k] + Q[k][j]*Q[k][j])

        if loss <= 0.001:
            break
        if epoch_i % 100 == 0:
            print("Epoch %d error: %.6f" % (epoch_i, loss))

    return P, Q

if __name__ == "__main__":
    file_name = "./data"
    data = load_data(file_name)
    P, Q = gra_descent(data, 20, 0.001, 1)
    print(P)
    print(Q)
    print(P*Q)