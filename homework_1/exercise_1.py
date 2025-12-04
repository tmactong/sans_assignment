import numpy as np

matrix_a = np.array([
    [4, 3, 2, 1],
    [3, 1, 4, 2]
])

matrix_b = np.array([
    [4, 3, 2],
    [3, 1, 5],
    [1, 4, 7],
    [5, 2, 6],
    [3, 4, 2]
])


y_a = np.array([
    [2],
    [1]
])

y_b = np.array([
    [1], [2], [1], [5], [4]
])

def calculate_pseudo_inverse_of_matrix_a():
    U, S, V = np.linalg.svd(matrix_a)
    S = np.diag(S)
    pseudo_inverse = V[0:2].T @ np.linalg.inv(S) @ U.T
    print(f'pseudo inverse of matrix A is {pseudo_inverse}')
    x = pseudo_inverse @ y_a
    print(f'Ax is {matrix_a @ x}')
    # print(f'A @ pseudo inverse = {matrix_a @ pseudo_inverse}')
    print(f'result is {matrix_a.T @ (y_a - matrix_a @ x)}')

def calculate_pseudo_inverse_of_matrix_b():
    U, S, V = np.linalg.svd(matrix_b)
    S = np.diag(S)
    pseudo_inverse = V.T @ np.linalg.inv(S) @ U[:, :3].T
    print(f'pseudo inverse is {pseudo_inverse}')
    print(f'pseudo_inverse @ matrix_b = {pseudo_inverse @ matrix_b}')
    x = pseudo_inverse @ y_b
    print(f'x is {x}')
    print(f'Bx is {matrix_b @ x}')
    Bx = matrix_b @ x
    print(f'B^t @ (c - Bx) = {matrix_b.T @ (y_b - Bx)}')

def main():
    calculate_pseudo_inverse_of_matrix_a()
    # calculate_pseudo_inverse_of_matrix_b()


if __name__ == "__main__":
    main()
