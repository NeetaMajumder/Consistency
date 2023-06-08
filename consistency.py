
import numpy as np

# Pairwise comparison matrix
pairwise_matrix = np.array([[1, 1/5, 2],
                            [5, 1, 5],
                            [1/2, 1/5, 1]])

# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)

# Find the index of the principal eigenvalue
principal_eigenvalue_index = np.argmax(np.abs(eigenvalues))

# Get the eigenvector associated with the principal eigenvalue
principal_eigenvector = eigenvectors[:, principal_eigenvalue_index]

# Normalize the eigenvector to obtain the weightage or priority vector
weightage_vector = principal_eigenvector / np.sum(principal_eigenvector)

# Number of criteria or alternatives being compared
n = pairwise_matrix.shape[0]

# Calculate the principal eigenvalue
principal_eigenvalue = np.abs(eigenvalues[principal_eigenvalue_index])

# Calculate the consistency index (CI)
CI = (principal_eigenvalue - n) / (n - 1)

# Random index (RI) for matrix size
RI = {3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}  

# Calculate the consistency ratio (CR)
CR = CI / RI[n]

print("Weightage or Priority Vector:", weightage_vector)
print("Consistency Index (CI):", CI)
print("Consistency Ratio (CR):", CR)



