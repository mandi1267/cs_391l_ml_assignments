
import numpy as np

def hw1FindEigendigits(training_data):
    """
    Computes the mean of the training data and an NxM matrix, with each column as an an eigenvector of the training data.
    N = number of features in the original training data, M = number of training samples. The eigenvectors are
    normalized to have length 1 and are sorted by their respective eigenvalues in descending order. Assumes that N >= M.

    Args:
        training_data (2D numpy array): Array containing training data. Each
            sample corresponds to one column.

    Returns:
        Tuple containing vector of length N (number of pixels) containing mean
        value of training samples for each pixel and NxM matrix containing the
        M highest eighenvectors (sorted in descending order by their
        eigenvalue), where M is the number of training samples. Each column is
        an eigenvector.
    """

    # Compute the mean of each feature (will give vector of length N)
    mean_vec = np.mean(training_data, axis=1)

    mean_0_training_data = training_data - mean_vec[:, None]
    mean_0_training_data_transpose = np.transpose(mean_0_training_data)

    # Get the MxM matrix (training data has N rows, so the transpose will have M rows)
    m_by_m_mat = mean_0_training_data_transpose.dot(mean_0_training_data)

    # Get the eigen vectors of the MxM matrix
    # Because a matrix times its transpose is symmetric, we can use eigh
    eig_vals, eig_vecs = np.linalg.eigh(m_by_m_mat)

    # Each column in eig_vecs is an eigenvector
    # Sort the eigenvectors by their corresponding eigenvalues in descending
    # order
    # Transpose the eigenvector matrix (so each eigenvector is a row) and
    # add a column at the end corresponding to the eigenvalue
    eig_vals_and_vecs_matrix = np.column_stack((np.transpose(eig_vecs), eig_vals))

    # Sort the combined matrix by the last column and then remove the
    # eigenvalue column
    sorted_eig_vecs_and_vals = eig_vals_and_vecs_matrix[(-1 * eig_vals_and_vecs_matrix[:, -1]).argsort()]
    sorted_eig_vecs = sorted_eig_vecs_and_vals[:, :-1]

    # Transform the eigenvectors so that they are eigenvectors of the training
    # data
    sorted_eig_vecs_in_columns = np.transpose(sorted_eig_vecs)
    m_largest_eig_vecs_of_n_by_n_mat = mean_0_training_data.dot(sorted_eig_vecs_in_columns)

    # Normalize the eigenvectors to have length 1
    eig_vector_magnitudes = np.linalg.norm(m_largest_eig_vecs_of_n_by_n_mat, axis=0)
    normalized_m_largest_eig_vecs_of_n_by_n_mat = m_largest_eig_vecs_of_n_by_n_mat / (eig_vector_magnitudes[None, :])

    # Return a tuple of the mean value for each feature (pixel) and a matrix with
    return mean_vec, normalized_m_largest_eig_vecs_of_n_by_n_mat

def transformToEigenRepresentation(mean_feature_vec, eigenvecs_of_training, data_to_eval):
    """
    Transform the input data to the eigenvector representation by subtracting the mean of each feature
    and then taking the dot product with each eigenvector to get a new feature of dimension M (where M is the number of
    eigenvectors). This is performed as a matrix operation for every sample in the input data (each column of the input
    data).

    Args:
        mean_feature_vec (1D numpy array):      Vector containing the mean of each feature in the full-dimensional
                                                representation of the data.
        eigenvecs_of_training (2D numpy array): N x M matrix where each column is an eigenvector of the training data.
                                                N is the number of features in the original data representation (number
                                                of pixels) and M is the number of eigenvectors and the size of the
                                                lower-dimensional representation for the data. M must be less than or
                                                equal to N.
        data_to_eval (2D numpy array):          N x L array containing data to transform to lower-dimensional
                                                representation. Each column is one sample. L is the number of samples.
    Returns:
        M x L array with the input data transformed to the eigenvector representation, with each column as a
        transformed sample. There are M features for each transformed sample and L total samples.
    """
    mean_subtracted_data = data_to_eval - mean_feature_vec[:, None]
    return np.transpose(np.transpose(mean_subtracted_data).dot(eigenvecs_of_training))
