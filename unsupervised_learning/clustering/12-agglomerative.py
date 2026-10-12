#!/usr/bin/env python3
"""Agglomerative clustering with Ward linkage"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering and display the dendrogram

    X: numpy.ndarray (n, d) dataset
    dist: maximum cophenetic distance for all clusters
    Returns: clss numpy.ndarray (n,) of cluster indices
    """
    hier = scipy.cluster.hierarchy
    Z = hier.linkage(X, method='ward')
    clss = hier.fcluster(Z, t=dist, criterion='distance')
    plt.figure()
    hier.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clss
