
import numpy as np

def readMNISTData(data_file_name):
    """
    Read images from a file containing MNIST image data.

    Args:
        data_file_name (string):    Name of the file containing the MNIST images to read.
    Returns:
        Tuple containing the following:
            image data - 2D numpy array of size M x N, where M is the number of images and N is the number of pixels in
                each image. Each row of the array is one image.
            num_rows - number of rows in each image
            num columns - number of columns in each image
    """
    with open(data_file_name, 'rb') as data_file:

        # First 4 bytes are an unused magic number
        data_file.read(4)

        # Next 3 4-byte sequences are number of images, number of rows in each image, and number of columns in each
        # image, in that order
        num_images = int.from_bytes(data_file.read(4), 'big')
        num_rows = int.from_bytes(data_file.read(4), 'big')
        num_columns = int.from_bytes(data_file.read(4), 'big')

        # The size of a single image vector is the number of rows times one of the columns
        image_size = num_rows * num_columns

        # Next is the data for all of the images
        # Want to read it and then shape it into a numpy array with num_images rows
        raw_image_data = data_file.read()
        image_data = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((num_images, image_size))
        return image_data, num_rows, num_columns

def readMNISTLabels(labels_file_name):
    """
    Read labels from a file containing labels for MNIST images.

    Args:
        labels_file_name (string):  Name of the file that contain the label data for MNIST images.
    Returns:
        1D numpy array with each entry as a label for an MNIST image.
    """
    with open(labels_file_name, 'rb') as labels_file:

        # First 4 bytes are an unused magic number
        labels_file.read(4)

        # Next 4 bytes are the number of labels
        num_labels = int.from_bytes(labels_file.read(4), 'big')

        # Remaining bytes are the labels
        raw_label_data = labels_file.read()
        labels_data = np.frombuffer(raw_label_data, dtype=np.uint8)

        return labels_data
