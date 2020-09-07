# Used to sanity check the mnist reader file
# Reads an input file and outptus the number and size of images
# and displays some of the images with their labels


import sys
from display_utils import *
from mnist_reader import *

def main():
    if (len(sys.argv) < 3):
        print("Need to specify files for data and labels")
        return

    data_file_name = sys.argv[1]
    labels_file_name = sys.argv[2]

    img_data, num_rows_in_img, num_cols_in_img = readMNISTData(data_file_name)
    img_labels = readMNISTLabels(labels_file_name)

    print("Number of images: " + str(img_data.shape[0]))
    print("Number of rows in image: " + str(num_rows_in_img))
    print("Number of columns in image: " + str(num_cols_in_img))

    print("Number of training_labels: " + str(img_labels.shape[0]))

    num_images_to_display = 15

    for i in range(num_images_to_display):
        data = img_data[i, :]
        matrix_version = convertVectorToImage(data, num_rows_in_img, num_cols_in_img)
        displayImage(matrix_version, img_labels[i])

if __name__=="__main__":
    main()
