import numpy as np


matrix_a = np.array([
    [4, 3, 2, 1],
    [1, 2,1,3],
    [3,1,0,2],
    [2,3,4,5]
])

matrix_b = np.array([
    [5, 0, 2, -1],
    [0, 2, 0, -1],
    [2, 0, 1, 0],
    [-1, -1, 0, 2]
])

matrix_p = np.array([
    [1, 1 , -1],
    [1 ,2,0],
    [-1, 0, 4]
])


def trace():
    c = matrix_a @ matrix_b
    print(c)
    print(np.trace(c))
    print(np.trace(matrix_b @ matrix_a))

def hadamard_product():
    print(matrix_a * matrix_b)

def eigenvalues():
    print(np.linalg.eig(matrix_p))



if __name__ == '__main__':
    # trace()
    # hadamard_product()
    eigenvalues()