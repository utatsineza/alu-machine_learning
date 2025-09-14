#!/usr/bin/env python3
"""
This script demonstrates list comprehension with matrices in Python.

It defines a 3x6 matrix (list of lists) and extracts the "middle"
columns (index 2 and 3 from each row). The result is then printed.
"""

def main():
    """
    Extract the middle columns from a 3x6 matrix and print them.
    """
    # Define a 3x6 matrix
    matrix = [
        [1, 3, 9, 4, 5, 8],
        [2, 4, 7, 3, 4, 0],
        [0, 3, 4, 6, 1, 5]
    ]

    # Use list comprehension to extract columns with indices 2 and 3
    the_middle = [row[2:4] for row in matrix]

    # Print results
    print("The middle columns of the matrix are: {}".format(the_middle))


if __name__ == "__main__":
    main()

