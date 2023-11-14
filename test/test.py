import numpy as np

arr = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]])
arr = np.minimum(arr, 9)
arr = np.maximum(arr, 3)
print(arr)
