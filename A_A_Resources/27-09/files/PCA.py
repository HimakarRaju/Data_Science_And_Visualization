''' Flow chart for the code
Start
  |
  v
Input: Dataset
  |
  v
Step 1: Mean Center the Data (Subtract Mean of Each Feature)
  |
  v
Step 2: Compute Covariance Matrix
  |
  v
Step 3: Compute Eigenvalues and Eigenvectors of Covariance Matrix
  |
  v
Step 4: Sort Eigenvalues (Pick top components)
  |
  v
Step 5: Project Data onto Principal Components
  |
  v
Output: Transformed Dataset with Reduced Dimensions
  |
  v
End



''''
def mean_center_data(data):
    mean_centered = []
    for i in range(len(data[0])):  # Loop over columns (features)
        mean = sum(row[i] for row in data) / len(data)
        mean_centered.append([row[i] - mean for row in data])
    return list(zip(*mean_centered))  # Return transposed

def covariance_matrix(data):
    N = len(data[0])
    cov_matrix = [[0] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            cov_matrix[i][j] = sum(data[i][k] * data[j][k] for k in range(len(data))) / (len(data[0]) - 1)

    return cov_matrix

def eigen_decomposition(matrix):
    # Simple power iteration method to find the largest eigenvalue and eigenvector
    n = len(matrix)
    eigenvector = [1] * n
    for _ in range(100):  # Iteration limit
        next_vector = [sum(matrix[i][j] * eigenvector[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x ** 2 for x in next_vector))
        eigenvector = [x / norm for x in next_vector]
    eigenvalue = sum(matrix[i][i] * eigenvector[i] for i in range(n))  # Approx eigenvalue
    return eigenvalue, eigenvector

def pca(data, n_components=1):
    mean_centered_data = mean_center_data(data)
    cov_matrix = covariance_matrix(mean_centered_data)

    # Perform eigen decomposition on the covariance matrix
    eigenvalue, eigenvector = eigen_decomposition(cov_matrix)
    
    # Project data onto the principal component
    projected_data = [[sum(row[i] * eigenvector[i] for i in range(len(row))) for row in data]]

    return projected_data[:n_components]

# Example usage
data = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]]
reduced_data = pca(data, n_components=1)
print(f"Projected Data: {reduced_data}")
