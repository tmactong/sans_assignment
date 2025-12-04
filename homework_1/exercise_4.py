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


def calculate_svd_of_matrix(m):
    U, S, V = np.linalg.svd(m)
    S = np.diag(S)
    print(V @ V.T)


def rank_one_approximation(m):
    U, S, V = np.linalg.svd(m)
    s1 = S[0]
    print(s1 * U[:,:1] @ V[0:1])

def rank_two_approximation(m):
    U, S, V = np.linalg.svd(m)
    print(S[0] * U[:,:1] @ V[0:1] + S[1] * U[:,1:2] @ V[1:2])

def frobenius_norm(m):
    _, S, _ = np.linalg.svd(m)
    print(np.sqrt(np.sum(np.square(S))))

def main():
    # calculate_svd_of_matrix(matrix_b)
    # rank_one_approximation(matrix_b)
    # rank_two_approximation(matrix_b)
    frobenius_norm(matrix_b)


if __name__ == "__main__":
    main()
