import numpy as np

matrix_a = np.array([
    [4, 3, 2, 1],
    [3, 1, 4, 2]
])


def validate_svd(matrix: np.ndarray):
    U, singular_values, V = np.linalg.svd(matrix_a)
    print('singular values:', singular_values)
    S = np.diag(singular_values)
    print(f'{matrix} vs {U @ S @ V[0:2]}')