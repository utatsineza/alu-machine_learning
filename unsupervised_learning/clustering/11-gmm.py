#!/usr/bin/env python3
"""GMM with sklearn"""
import sklearn.mixture


def gmm(X, k):
    """Calculate a GMM from a dataset using sklearn

    Returns: pi, m, S, clss, bic
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
