#!/usr/bin/env python3
"""K-means with sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """Perform K-means on a dataset using sklearn

    Returns: C, clss
    """
    km = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return km.cluster_centers_, km.labels_
