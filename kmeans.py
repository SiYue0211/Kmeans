#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 17:36
# @Author  : dawn
# @File    : kmeans.py

import numpy as np

class KMeansClassifier():
    def __init__(self, k=3, initCent='random', max_iter=500):
        self._k = k
        self._initCent = initCent
        self._max_iter = max_iter
        # 记录每一个点属于哪个聚类中心，和与聚类中心的误差, 形状【所有数据的长度，2】
        self._clusterAssment = None
        # 记录所有点属于哪个聚类中心，形状【所有数据的长度，1】
        self._label = None
        # 记录所有点与它所属的聚类中心的误差， 形状【所有数据长度， 1】
        self._see = None

    def _calEDist(self, A, B):
        '''
        :param A:
        :param B:
        :return:计算A和B之间的欧拉距离
        '''
        return np.math.sqrt(sum(np.power(A - B, 2)))

    def _randCent(self, data):
        randIdex = np.random.randint(0, data.shape[0], size=(self._k))
        centroids = data[randIdex, :]
        return centroids


    def fit(self, data):
        if not isinstance(data, np.ndarray) or isinstance(data, np.matrixlib.defmatrix.matrix):
            try:
                data = np.asarray(data)
            except:
                raise TypeError("numpy.ndarray resuired for data_X")
        m = data.shape[0]
        self._clusterAssment = np.zeros((m, 2))
        self._label = np.zeros((m, 1))

        if self._initCent == 'random':
            self._centroids = self._randCent(data)

        clusterChanged = True
        for _ in range(self._max_iter):
            clusterChanged = False
            for i in range(m):
                minDist = np.inf # 首先将mindist设成一个无穷大的数
                minIndex = -1
                for j in range(self._k):
                    arrA = self._centroids[j, :]
                    arrB = data[i, :]
                    disIJ = self._calEDist(arrA, arrB)

                    if disIJ < minDist:
                        minDist = disIJ
                        minIndex = j
                if self._clusterAssment[i, 1] != minDist or self._clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    self._clusterAssment[i, :] = minIndex, minDist

            if not clusterChanged:
                break;

            # 更新质心
            for i in range(self._k):
                k_index = np.where(self._clusterAssment[:, 0] == i)[0]
                k_data = data[k_index, :]
                self._centroids[i, :] = np.mean(k_data, axis=0)

        self._label = self._clusterAssment[:, 0]
        self._see = sum(self._clusterAssment[:, 1])






















