import math
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

def getMnistData(train_set_size, test_set_size):
    mnist = tf.keras.datasets.mnist  # Object of the MNIST dataset
    (training_data, train_labels), (test_data, test_labels) = mnist.load_data()  # Load data

    print(training_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    training_data = tf.reshape(training_data, (-1, 784)).numpy()
    test_data = tf.reshape(test_data, (-1, 784)).numpy()

    all_train_data = getRandomSubsetOfData(training_data, train_labels, train_set_size)
    all_test_data = getRandomSubsetOfData(test_data, test_labels, test_set_size)

    return (all_train_data, all_test_data)

def displayCNNAccuracy(cnn_accuracy_by_sample_size, num_train, num_test):
    # print("CNN test accuracy on " + str(num_test) + " samples trained using " + str(num_train) + " samples: " + str(cnn_accuracy))
    # TODO what input?

    train_sizes = [train_size for train_size in cnn_accuracy_by_sample_size.keys()]
    accuracies = [cnn_accuracy_by_sample_size[train_size] for train_size in train_sizes]


    plt.plot(train_sizes, accuracies)

    plt.xlabel("Training set size")
    plt.ylabel("Accuracy rate")
    plt.title("CNN Classification Accuracy by Training Set Size")
    plt.legend()
    plt.show()


def executeCNNClassification(train_data, test_data):

    full_train_set_size = train_data[0].shape[0]
    print("Full train")
    print(full_train_set_size)
    train_set_sizes = [int(0.001 * i * full_train_set_size) for i in range(5, 50, 5)]
    train_set_sizes.extend([int(0.01 * i * full_train_set_size) for i in range(5, 101, 5)])
    print(train_set_sizes)
    cnn_accuracy_by_train_set_size = {}

    cnn_epochs = 10
    for train_set_size in train_set_sizes:
        print("Train set size " + str(train_set_size))
        cnn_classifier = CnnClassifier()
        train_images_subset, train_labels_subset = getRandomSubsetOfData(train_data[0], train_data[1], train_set_size)
        cnn_classifier.trainModel(train_images_subset, train_labels_subset, cnn_epochs)
        cnn_accuracy = cnn_classifier.testModel(test_data[0], test_data[1])
        cnn_accuracy_by_train_set_size[train_set_size] = cnn_accuracy


    displayCNNAccuracy(cnn_accuracy_by_train_set_size, train_data[0].shape[0], test_data[0].shape[0])

def plotGanTrainLosses(training_losses_by_iteration, show_plt=True, name_prefix=None):

    iterations_nums = [iter for iter in training_losses_by_iteration.keys()]

    generator_fool_loss = [training_losses_by_iteration[iteration].generator_train_realness_loss for iteration in iterations_nums]
    discriminator_fool_loss = [training_losses_by_iteration[iteration].discriminator_fool_loss for iteration in iterations_nums]

    plt.plot(iterations_nums, discriminator_fool_loss, label="Discriminator loss")
    plt.plot(iterations_nums, generator_fool_loss, label="Generator loss")

    plt.xlabel("Iteration number")
    plt.ylabel("Loss")
    plt.title("Discriminator and Generator Losses for Real/Synthetic Differentiation")
    plt.legend()

    if (name_prefix != None):
        plt.savefig((name_prefix + "_realness_loss.png"))
    plt.show(block=show_plt)
    plt.close()


    generator_class_loss = [training_losses_by_iteration[iteration].generator_class_loss for iteration in iterations_nums]
    discriminator_class_loss = [training_losses_by_iteration[iteration].discriminator_class_loss for iteration in iterations_nums]

    plt.plot(iterations_nums, discriminator_class_loss, label="Discriminator loss")
    plt.plot(iterations_nums, generator_class_loss, label="Generator loss")

    plt.ylim((0, 2))
    plt.xlabel("Iteration number")
    plt.ylabel("Loss")
    plt.title("Discriminator and Generator Losses for Digit Classification")
    plt.legend()
    if (name_prefix != None):
        plt.savefig((name_prefix + "_classification_loss.png"))
    plt.show(block=show_plt)
    plt.close()


def plotImageGrid(images, latent_labels, title, image_grid_filename, show_plt=True):
    grid_dim = math.trunc(pow(len(images), 0.5))

    for i in range(pow(grid_dim, 2)):
        # define subplot
        ax = pyplot.subplot(grid_dim, grid_dim, 1 + i)
        image = images[i]
        image = tf.reshape(image, [28, 28])
        plt.imshow(normalizeToIntRange(image))

        plt.axis('off')
        plt.subplots_adjust(hspace=0.5)
        ax.set_title("label: " + str(latent_labels[i]), fontsize=10)
    fig = plt.gcf()
    fig.set_size_inches(4, 5)
    plt.suptitle(title)
    if (image_grid_filename != None):
        plt.savefig(image_grid_filename)
    plt.show(block=show_plt)
    plt.close()

def plotGeneratedImagesByEpoch(generated_images_by_iteration, latent_labels):
    num_epochs = 75

    iters_per_epoch = len(generated_images_by_iteration) / num_epochs

    images_by_epoch = {}

    for iter, images in generated_images_by_iteration.items():
        epoch = math.trunc(iter / iters_per_epoch)
        # print(epoch)
        # print(iter)
        # print(iters_per_epoch)
        images_by_epoch[epoch] = images

    print("Images per iter " )
    print(len(images_by_epoch[0]))

    for epoch, images in images_by_epoch.items():
        plotImageGrid(images, latent_labels, ("Generated Images at end of Epoch " + str(epoch)), ("gen_images_at_epoch_" + str(epoch) + ".png"), False)

def plotImage(image, title, show_plt=True, filename=None):
    plt.title(title)
    plt_image = tf.reshape(image, [28, 28])
    plt.imshow(plt_image)
    if filename != None:
        plt.savefig(filename)
    plt.show(block=show_plt)
    plt.close()

def plotMisclassifiedImages(misclassified_images, true_labels, generated_labels, show_plt=True, img_file_prefix=None):
    max_num_to_plot = 30

    for i in range(min(max_num_to_plot, (misclassified_images.shape[1]))):
        image = misclassified_images[0, i, :]
        true_label = true_labels[i]
        generated_label = generated_labels[i]


        title = "Misclassified image " + str(i) + "\nTrue label: " + str(true_label) + "\nClassification label: " + str(generated_label)
        filename = None
        if (img_file_prefix != None):
            filename = img_file_prefix + "_" + str(i) + ".png"
        plotImage(image, title, show_plt, filename)


def classifyWithGanDiscriminator(discriminator, test_data):
    print("Classifying")
    test_images = test_data[0]
    test_images = normalizeToFloatRange(test_images)
    test_labels = test_data[1]

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

    joblib.dump((accuracy, misclassified_img_correct_label_subset, misclassified_data_subset, misclassified_img_act_label_subset), misclassified_fpath)

    plotMisclassifiedImages(misclassified_data_subset, misclassified_img_correct_label_subset, misclassified_img_act_label_subset, show_plt=True, img_file_prefix=time_str)

    # TODO Find some misclassified examples and plot

    print("GAN Discriminator classification accuracy " + str(accuracy))

def trainGAN(generator, discriminator, train_data, latent_space_dim, num_classes):
    print("Training GAN")
    full_gan = Gan(generator, discriminator)
    # TODO create latent data and classes to compute resutls for
    # num_epochs = 5
    num_epochs = 75
    batch_size = 64

    constistent_samples_num = 25

    train_data = (normalizeToFloatRange(train_data[0]), train_data[1])

    test_latent_rep, test_latent_labels = createGeneratorInput(latent_space_dim, constistent_samples_num, num_classes)
    loss_by_iter, generated_images_by_iter = trainGanByBatches(generator, discriminator, full_gan, train_data, latent_space_dim, test_latent_rep, test_latent_labels,
                      num_epochs, batch_size)
    time_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    losses_and_imgs_fpath = "gan_final_output" + time_str + ".pkl"

    joblib.dump((loss_by_iter, generated_images_by_iter, test_latent_labels), losses_and_imgs_fpath)

    plotGanTrainLosses(loss_by_iter, show_plt=False, name_prefix=("loss_" + time_str + "_"))
    plotGeneratedImagesByEpoch(generated_images_by_iter, test_latent_labels)

    print("Done training GAN")
    return loss_by_iter, generated_images_by_iter

def createGeneratorAndDiscriminator(num_classes, latent_space_dim):
    discriminator = Discriminator(num_classes)
    generator = Generator(latent_space_dim, num_classes)

    return (generator, discriminator)

def executeAssign6(train_data, test_data):

    # executeCNNClassification(train_data, test_data)

    num_classes = 10
    latent_space_dim = 100
    (generator, discriminator) = createGeneratorAndDiscriminator(num_classes, latent_space_dim)
    discriminator.model.summary()
    (losses, images) = trainGAN(generator, discriminator, train_data, latent_space_dim, num_classes)

    # gan_discriminator = trainGanForDiscrimination(train_data)
    # digit_discriminator = modifyGanDiscriminatorForClassification(gan_discriminator, train_data)
    classifyWithGanDiscriminator(discriminator, test_data)

def plotSavedLossesAndImages():
    results_file_name = "gan_final_output2020-12-10T19:53:07.pkl"

    (loss_by_iter, generated_images_by_iter) = joblib.load(results_file_name)

    # (loss_by_iter, generated_images_by_iter, latent_labels) = joblib.load(results_file_name)
    latent_labels = [7, 7, 1, 1, 2, 2, 8, 9, 3, 8, 4, 0, 6, 2, 6, 7, 3, 3, 6, 1, 8, 2, 1, 4, 8] # TODO

    # plotGanTrainLosses(loss_by_iter)
    plotGeneratedImagesByEpoch(generated_images_by_iter, latent_labels)

def plotSavedMisclassifiedImages():
    filename = "misclassified_imgs_2020-12-10T19:53:11.pkl"
    (misclassified_img_correct_label_subset, misclassified_data_subset, misclassified_img_act_label_subset) = joblib.load(filename)

    plotMisclassifiedImages(misclassified_data_subset, misclassified_img_correct_label_subset, misclassified_img_act_label_subset, True, "misclassified_img_plots_")


def getCmdLineArgs():

    """
    Set up the command line argument options and return the arguments used.

    Returns:
        Command line arguments data structure.
    """
    default_training_data_file_name = "/home/amanda/Downloads/train-images-idx3-ubyte.gz"
    default_training_label_file_name = "/home/amanda/Downloads/train-labels-idx1-ubyte.gz"

    default_test_data_file_name = "/home/amanda/Downloads/t10k-images-idx3-ubyte.gz"
    default_test_label_file_name = "/home/amanda/Downloads/t10k-labels-idx1-ubyte.gz"

    default_train_set_size = 20000 # TODO
    # default_train_set_size = 64
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

    train_data, test_data = getMnistData(train_set_size, test_set_size)

    # train_data = getMNistTrain(train_image_file_name, train_label_file_name, train_set_size)
    # test_data = getMNistTest(test_image_file_name, test_label_file_name, test_set_size)

    executeAssign6(train_data, test_data)
    # plotSavedMisclassifiedImages()
    # plotSavedLossesAndImages()


