#!/usr/bin/env python3
"""
This script demonstrates basic list slicing in Python.

It creates a list of integers and extracts:
- the first two elements,
- the last five elements,
- and elements from index 1 through 5 (2nd through 6th).

The results are then printed in a readable format.
"""

def main():
    """
    Perform slicing operations on a list and print the results.
    """
    arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]

    # Get the first two numbers (indexes 0 and 1)
    arr1 = arr[:2]

    # Get the last five numbers (indexes -5 through end)
    arr2 = arr[-5:]

    # Get numbers from index 1 through 5 (2nd through 6th elements)
    arr3 = arr[1:6]

    # Print results
    print("The first two numbers of the array are: {}".format(arr1))
    print("The last five numbers of the array are: {}".format(arr2))
    print("The 2nd through 6th numbers of the array are: {}".format(arr3))


if __name__ == "__main__":
    main()

