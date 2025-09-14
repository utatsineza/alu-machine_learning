import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two numpy arrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray): Second input array.
        axis (int): Axis along which to concatenate (default 0).

    Returns:
        numpy.ndarray: Concatenated array.
    """
    return np.concatenate((mat1, mat2), axis=axis)

