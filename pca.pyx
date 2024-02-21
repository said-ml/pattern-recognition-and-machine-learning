# File: pca.pyx
import numpy as np
cimport numpy as np

cpdef np.ndarray[np.float64_t, ndim=2] pca_transform(np.ndarray[np.float64_t, ndim=2] X, int n_components):
    cdef np.ndarray[np.float64_t, ndim=2] centered_X
    cdef np.ndarray[np.float64_t, ndim=2] cov_matrix
    cdef np.ndarray[np.float64_t, ndim=2] eigenvalues
    cdef np.ndarray[np.float64_t, ndim=2] eigenvectors
    cdef np.ndarray[np.float64_t, ndim=2] sorted_eigenvectors
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]

    # Center the data
    centered_X = X - np.mean(X, axis=0)

    # Calculate the covariance matrix
    cov_matrix = np.dot(centered_X.T, centered_X) / n_samples

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors based on eigenvalues
    idx = eigenvalues.argsort()[::-1]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Select top n_components eigenvectors
    sorted_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Project data onto the principal components
    transformed_X = np.dot(centered_X, sorted_eigenvectors)

    return transformed_X
