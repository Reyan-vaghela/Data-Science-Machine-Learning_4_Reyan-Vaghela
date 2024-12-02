import numpy as np

def matrix_factorization_svd(matrix):

    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    S_matrix = np.diag(S)
    
    return U, S_matrix, VT

if __name__ == "__main__":
    A = np.array([[2, 0, 0],
                  [3, 0, 0],
                  [0, 2, 0],
                  [0, 0, 0],
                  [0, 1, 0]])

    U, S, VT = matrix_factorization_svd(A)

    print("Original Matrix (A):")
    print(A)
    print("\nU Matrix:")
    print(U)
    print("\nS Matrix:")
    print(S)
    print("\nVT Matrix:")
    print(VT)

    reconstructed_A = np.dot(U, np.dot(S, VT))
    
    print("\nReconstructed Matrix (A):")
    print(reconstructed_A)