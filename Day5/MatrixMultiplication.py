# # a = [[2, 3], [4, 3]]
# # b = [[6, 7], [8, 9]]
# # c = [[0, 0], [0, 0]]

# # for i in range(len(a)):
# #     for j in range(len(b[0])):
# #         for k in range(len(b)):
# #             c[i][j] += a[i][k] * b[k][j]
# # for result in c:
# #     print(result)


# # matrix multiplication?
# def matrix_multiply(A, B):

#     # num_rows_A = len(A)
#     # num_cols_A = len(A[0])
#     # num_rows_B = len(B)
#     # num_cols_B = len(B[0])

#     # Check if the matrices can be multiplied
#     if len(A[0]) != len(B):
#         raise ValueError("Matrices cannot be multiplied")

#     # Create the result matrix
#     C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

#     # Perform matrix multiplication
#     for i in range(len(A)):
#         for j in range(len(B[0])):
#             for k in range(len(A[0])):  # or 'num_rows_B'
#                 C[i][j] += A[i][k] * B[k][j]

#     return C


# # Define two matrices
# A = [[1, 2], [3, 4]]
# B = [[5, 6], [7, 8]]

# # Perform matrix multiplication
# C = matrix_multiply(A, B)

# print(C)


# Code given by sir
def recur_mul(A, B):
    # check if matrices can be multiplied
    if len(A[0]) != len(B):
        raise ValueError("Invalid matrix dimensions")

    # initialize result matrix with zeros
    result = [[0 for j in range(len(B[0]))] for i in range(len(A))]

    # recursive multiplication of matrices
    def multiply(A, B, result, i, j, k):
        if i >= len(A):
            return
        if j >= len(B[0]):
            return multiply(A, B, result, i+1, 0, 0)
        if k >= len(B):
            return multiply(A, B, result, i, j+1, 0)
        result[i][j] += A[i][k] * B[k][j]
        multiply(A, B, result, i, j, k+1)

    # perform matrix multiplication
    multiply(A, B, result, 0, 0, 0)
    return result


# example usage
A = [[12, 7, 3], [4, 5, 6], [7, 8, 9]]
B = [[5, 8, 1, 2], [6, 7, 3, 0], [4, 5, 9, 1]]

result = recur_mul(A, B)
for row in result:
    print(row)
