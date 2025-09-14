#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for row in mat1:
        new_row = []
        for j in range(len(mat2[0])):
            s = sum(row[i] * mat2[i][j] for i in range(len(mat2)))
            new_row.append(s)
        result.append(new_row)
    return result

