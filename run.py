#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/13 17:37
# @Author  : dawn
# @File    : run.py

import numpy as np
import pandas as pd
from kmeans import KMeansClassifier
import matplotlib.pyplot as plt

# 加载数据集
def loadDataset(infile):
    df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
    return np.array(df).astype(np.float)

if __name__ == "__main__":
    data_X = loadDataset('data/testSet.txt')
    k = 3
    clf = KMeansClassifier()
    clf.fit(data_X)

    cent = clf._centroids
    label = clf._label
    see = clf._see
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']

    for i in range(k):
        index = np.where(label == i)[0]
        x = data_X[index, 0]
        y = data_X[index, 1]

        for j in range(len(x)):
            plt.text(x[j], y[j], str(i), color=colors[i], fontdict={'weight': 'bold', 'size': 6})

        plt.scatter(cent[i, 0], cent[i, 1], marker='x', color=colors[i], linewidths=7)

    plt.title("SSE={:.2f}".format(see))
    plt.axis([-7, 7, -7, 7])
    outname = "./result/k_clusters" + str(k) + ".png"
    plt.savefig(outname)
    plt.show()














