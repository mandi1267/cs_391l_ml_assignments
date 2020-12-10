from generator import *
from gan_discriminator import *
from mnist_reader import *
from cnn_classifier import *
import argparse
import tensorflow as tf
# from IPython import display

import joblib

from ac_gan_library import *

def getRandomSubsetOfData(data, labels, number_of_samples_to_return):
    """
    Get a random subset of the input data and corresponding labels.

    Args:
        data (2D numpy array):              Data to get the subset of. This is a L X P array, where P is the number of
                                            features in the data and L is the total number of samples. Each row is
                                            one sample.
        labels (1D numpy array):            Labels for the input data. Has L entries, where the first entry corresponds
                                            to the first column (sample) in the data, and so on.
        number_of_samples_to_return (int):  Number of total samples to return. Assumed to be less than or equal to the
                                            number of samples in the input data.
    Returns:
        A tuple of the following:
            number_of_samples_to_return x P array with a random selection of the input data, drawn without replacement.
                Each row is a sample
            number_of_samples_to_return length vector with the labels corresponding to the selected samples returned.
    """
    # Get the indices of the samples to return as a random sample
    indices = np.random.choice(labels.size, number_of_samples_to_return, replace=False)

    # Get the labels and data that corresponds to the randomly selected indices
    label_subset = labels[indices]
    data_subset = data[indices, :]
    return (data_subset, label_subset)


def getMNistTrain(mnist_train_data_file, mnist_train_label_file, train_set_size):
    # Training data is integers in the range 0 to 255
    training_data, num_rows_in_img, num_cols_in_img = readMNISTData(mnist_train_data_file)
    print(training_data)
    training_labels = readMNISTLabels(mnist_train_label_file)
    return getRandomSubsetOfData(training_data, training_labels, train_set_size)


def getMNistTest(mnist_test_data_file, mnist_test_label_file, test_set_size):
    test_data, num_rows_in_img, num_cols_in_img = readMNISTData(mnist_test_data_file)
    test_labels = readMNISTLabels(mnist_test_label_file)
    return getRandomSubsetOfData(test_data, test_labels, test_set_size)



def displayCNNAccuracy(cnn_accuracy, num_train, num_test):
    print("CNN test accuracy on " + str(num_test) + " samples trained using " + str(num_train) + " samples: " + str(cnn_accuracy))
    # TODO what input?
    pass


def executeCNNClassification(train_data, test_data):

    cnn_epochs = 10

    cnn_classifier = CnnClassifier()
    cnn_classifier.trainModel(train_data[0], train_data[1], cnn_epochs)
    cnn_accuracy = cnn_classifier.testModel(test_data[0], test_data[1])

    displayCNNAccuracy(cnn_accuracy, train_data[0].shape[0], test_data[0].shape[0])

    return cnn_accuracy

def plotGanTrainLosses(training_losses_by_iteration):
    pass

def plotImagesByIteration(generated_images_by_iteration):
    pass

#
# def trainGanForDiscrimination(train_data):
#     BUFFER_SIZE = 60000
#     BATCH_SIZE = 256
#     train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#
#     # TODO
#     num_iterations = 50 # TODO
#     num_images_per_iter_plot = 2 # TODO
#     num_images_generator_produce = 30
#     training_losses = []
#     generator = Generator()
#     discriminator = GanDiscriminator()
#
#     gan_input_dim = 100
#     seed = tf.random.normal([num_images_generator_produce, gan_input_dim])
#
#     batch_size = 256
#
#     generated_images_by_iteration = []
#     for i in range(num_iterations):
#         for image_batch in train_dataset:
#             discriminator_train_loss, generator_train_loss = runGanIteration(generator, discriminator, image_batch,
#                                                                          seed, gan_input_dim, num_images_generator_produce, batch_size)
#
#         # display.clear_output(wait=True)
#         generate_and_save_images(generator,
#                                  i + 1,
#                                  seed)
#         training_losses.append((discriminator_train_loss, generator_train_loss))
#
#         test_gan_input = tf.random.normal([num_images_per_iter_plot, gan_input_dim])
#
#         generated_images_by_iteration.append([generator.generateData(test_gan_input) for i in range(num_images_per_iter_plot)])
#
#     # display.clear_output(wait=True)
#     generate_and_save_images(generator,
#                              num_iterations,
#                              seed)
#     plotGanTrainLosses(training_losses)
#     plotImagesByIteration(generated_images_by_iteration)
#
#     return discriminator


# def modifyGanDiscriminatorForClassification(discriminator, train_data):
#     pass

def classifyWithGanDiscriminator(discriminator, test_data):
    print("Classifying")
    test_images = test_data[0]
    test_labels = test_data[1]

    test_images = normalizeToFloatRange(test_images)
    accuracy = discriminator.getClassificationAccuracy(test_images, test_labels)

    classification = discriminator.getClassification(test_images)
    # print(classification)
    # print(test_labels)
    correct_label_vec = test_labels - classification
    misclassified_indices = np.nonzero(correct_label_vec)
    # print(misclassified_indices)

    print("Number of misclassified " + str(len(misclassified_indices[0])))
    misclassified_img_correct_label_subset = test_labels[misclassified_indices]
    # print(misclassified_img_correct_label_subset)
    misclassified_data_subset = test_images[misclassified_indices, :]
    misclassified_img_act_label_subset = classification[misclassified_indices]


    time_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    misclassified_fpath = "misclassified_imgs_" + time_str + ".pkl"

    joblib.dump((misclassified_img_correct_label_subset, misclassified_data_subset, misclassified_img_act_label_subset), misclassified_fpath)

    # TODO Find some misclassified examples and plot

    print("GAN Discriminator classification accuracy " + str(accuracy))

def trainGAN(generator, discriminator, train_data, latent_space_dim, num_classes):
    print("Training GAN")
    full_gan = Gan(generator, discriminator)
    # TODO create latent data and classes to compute resutls for
    # num_epochs = 5
    num_epochs = 50
    batch_size = 64

    constistent_samples_num = 25

    train_data = (normalizeToFloatRange(train_data[0]), train_data[1])

    test_latent_rep, test_latent_labels = createGeneratorInput(latent_space_dim, constistent_samples_num, num_classes)
    loss_by_iter, generated_images_by_iter = trainGanByBatches(generator, discriminator, full_gan, train_data, latent_space_dim, test_latent_rep, test_latent_labels,
                      num_epochs, batch_size)
    time_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    losses_and_imgs_fpath = "gan_final_output" + time_str + ".pkl"

    joblib.dump((loss_by_iter, generated_images_by_iter), losses_and_imgs_fpath)

    print("Done training GAN")
    return loss_by_iter, generated_images_by_iter

def createGeneratorAndDiscriminator(num_classes, latent_space_dim):
    discriminator = Discriminator(num_classes)
    generator = Generator(latent_space_dim, num_classes)

    return (generator, discriminator)

def executeAssign6(train_data, test_data):

    executeCNNClassification(train_data, test_data)

    num_classes = 10
    latent_space_dim = 100
    (generator, discriminator) = createGeneratorAndDiscriminator(num_classes, latent_space_dim)
    (losses, images) = trainGAN(generator, discriminator, train_data, latent_space_dim, num_classes)

    # plotImages()
    # plotImagesByIteration()
    plotGanTrainLosses(losses)

    # gan_discriminator = trainGanForDiscrimination(train_data)
    # digit_discriminator = modifyGanDiscriminatorForClassification(gan_discriminator, train_data)
    classifyWithGanDiscriminator(discriminator, test_data)



def getCmdLineArgs():

    """
    Set up the command line argument options and return the arguments used.

    Returns:
        Command line arguments data structure.
    """
    default_training_data_file_name = "/Users/mandiadkins/Downloads/train-images.idx3-ubyte"
    default_training_label_file_name = "/Users/mandiadkins/Downloads/train-labels.idx1-ubyte"

    default_test_data_file_name = "/Users/mandiadkins/Downloads/t10k-images.idx3-ubyte"
    default_test_label_file_name = "/Users/mandiadkins/Downloads/t10k-labels.idx1-ubyte"

    # default_train_set_size = 20000 # TODO
    default_train_set_size = 1000
    default_test_set_size = 10000 # TODO
    # default_test_set_size = 64 # TODO

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train_image_file", default=default_training_data_file_name, help='Name of the file '\
        'containing the training images. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--train_label_file", default=default_training_label_file_name, help='Name of the file '\
        'containing the training labels. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--test_image_file", default=default_test_data_file_name, help='Name of the file '\
        'containing the test images. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--test_label_file", default=default_test_label_file_name, help='Name of the file'\
        'containing the test labels. Should be formatted based on the MNIST format described on the MNIST website.')
    arg_parser.add_argument("--train_set_size", type=int, default=default_train_set_size,
                            help='Number of samples to use for training')
    arg_parser.add_argument("--test_set_size", type=int, default=default_test_set_size,
                            help='Number of samples to use for testing')

    return arg_parser.parse_args()

if __name__ == '__main__':

    parser_results = getCmdLineArgs()

    train_image_file_name = parser_results.train_image_file
    train_label_file_name = parser_results.train_label_file
    test_image_file_name = parser_results.test_image_file
    test_label_file_name = parser_results.test_label_file
    train_set_size = parser_results.train_set_size
    test_set_size = parser_results.test_set_size

    train_data = getMNistTrain(train_image_file_name, train_label_file_name, train_set_size)
    test_data = getMNistTest(test_image_file_name, test_label_file_name, test_set_size)

    executeAssign6(train_data, test_data)


