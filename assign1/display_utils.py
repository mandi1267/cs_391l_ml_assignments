
import matplotlib.pyplot as plt
import numpy as np


def convertVectorToImage(image_vector, num_rows, num_columns):
    """
    Convert the vector, which is a 1D numpy array of length (num_rows * num_columns) to a 2D numpy array with num_rows
    rows and num_columns columns. Assumes vector is in row-major order.

    Args:
        image_vector (1D numpy array):  Unrolled (vector) representation of the image.
        num_rows (int):                 Number of rows in the image.
        num_columns (int):              Number of columns in the image.
    Returns:
        2D numpy array representation of the image.
    """
    return image_vector.reshape(num_rows, num_columns)

def displayImage(image_data, true_label=None, classified_label=None):
    """
    Display the given image and the labels associated with the image.

    Args:
        image_data (2D numpy array):    Image data as numpy array.
        true_label (int):               True label from the data set for the image.
        classified_label (int):         Optional classification label. If supplied, the both the true label and the
                                        label output by the classifier will be displayed. If this is not provided, only
                                        the true label will be displayed.
    """
    title_str = "Label: " + str(true_label)
    if (classified_label is not None):
        title_str = title_str + ", Classified Label: " + str(int(classified_label))
    plt.title(title_str)
    plt.imshow(image_data, interpolation='none')
    plt.show()

def displayEigenVector(num_rows_in_img, num_cols_in_img, eigen_vec):
    """
    Display the eigenvector as an image.

    Args:
        num_rows_in_img (int):      Number of rows in images.
        num_cols_in_img (int):      Number of columns in images.
        eigen_vec (1D numpy array): Eigenvector to display as an image.
    """
    eigen_vector_matrix = convertVectorToImage(eigen_vec, num_rows_in_img, num_cols_in_img)
    plt.title("Eigen vector")
    plt.imshow(eigen_vector_matrix, interpolation='none')
    plt.show()


def displayEigenVectorRepresentationOfImage(eigen_vec_img_representation, eigen_vector_matrix, num_rows_in_orig_img,
    num_cols_in_orig_img, mean_feature_vec):
    """
    Reconstruct and display an image from the eigenvector-space representation of the image and the eigenvector matrix
    and mean value of the pixels from which the eigenvectors were computed.

    Args:
        eigen_vec_img_representation (1D Numpy array):  Eigenvector-space representation of an image (reduced-feature image representation).
        eigen_vector_matrix (2D Numpy array):           N x M matrix with the top M eigenvectors of the training data as columns.
        num_rows_in_orig_img (int):                     Number of rows in the original image format.
        num_cols_in_orig_img (int):                     Number of columns in the original image format.
        mean_feature_vec (1D Numpy array):              Mean value for each pixel in the original training data set.
    """

    eig_pseudo_inv = np.linalg.pinv(eigen_vector_matrix)
    expanded_img_vec = (eigen_vec_img_representation @ eig_pseudo_inv) + mean_feature_vec
    eigen_vector_matrix = convertVectorToImage(expanded_img_vec, num_rows_in_orig_img, num_cols_in_orig_img)
    plt.title("Reconstructed image from eigenvector representation with " + str(eigen_vec_img_representation.size) + " features")
    plt.imshow(eigen_vector_matrix, interpolation='none')
    plt.show()

def plotGraph(x_values, y_values_dict, x_axis_label, y_axis_label, graph_title, display_color_str_map={}):
    """
    Plot a graph with the given data.

    Args:
        x_values (1D array/dict of str to 1D array):        X values to plot. Either a list of x values, or a dictionary
                                                            mapping the y label to the x values corresponding to that y
                                                            label.
        y_values (dict of string to 1D array):              Dictionary of y-value label to show in legend to the
                                                            corresponding y-values to plot. These will be plotted
                                                            against x_values.
        x_axis_label (string):                              Label for the x axis
        y_axis_label (string):                              Label for the y axis
        graph_title (string):                               Title for the graph
        display_color_str_map (dict of string to string):   Dictionary of y value label to (see y-values) to the format
                                                            string that should be used to plot the line for that y-data.
    """
    using_x_val_dict = type(x_values) is dict
    for label, y_values in y_values_dict.items():
        display_str = display_color_str_map.get(label, 'rD-')
        if (using_x_val_dict):
            plt.plot(x_values[label], y_values, display_str, label = label)
        else:
            plt.plot(x_values, y_values, display_str, label = label)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(graph_title)
    if (len(y_values_dict) > 1):
        plt.legend()
    plt.show()
