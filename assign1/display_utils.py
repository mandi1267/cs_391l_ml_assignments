
import matplotlib.pyplot as plt


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
