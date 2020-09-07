
import matplotlib.pyplot as plt

def convertImageToVector(image_array):
    # todo
    pass

def convertVectorToImage(image, num_rows, num_columns):
    return image.reshape(num_rows, num_columns)

def displayImage(image_data, true_label, classified_label=None):
    title_str = "Label: " + str(true_label)
    if (classified_label is not None):
        title_str = title_str + str(classified_label)
    plt.title(title_str)
    plt.imshow(image_data, interpolation='none')
    plt.show()
