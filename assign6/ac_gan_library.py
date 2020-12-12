"""
This file includes functions and classes related to the ACGAN and training it.
"""

import math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import RandomNormal
from matplotlib import pyplot
import os

import datetime

import joblib

def normalizeToFloatRange(img_data):
    """
    Normalize the image data to be in the range -1 to 1 from the range 0 to 255 and make sure it is floats.

    :param img_data:    Image data in the range 0 to 255

    :return: Image data in the range -1 to 1
    """
    img_data = img_data.astype('float32')
    # Scale the data from 0,255 to -1,1
    return (img_data - 127.5) / 127.5

def normalizeToIntRange(img_data):
    """
    Normalize the image data to be in the range 0 to 255 from the range -1 to 1 and make sure it is floats.

    :param img_data:    Image data in the range 0 to 255

    :return: Image data in the range -1 to 1
    """
    return img_data * 127.5 + 127.5


class Discriminator(object):
    """
    Discriminator class that can distinguish between real/synthetic images and between images of different classes.
    """

    def __init__(self, num_classes):
        # weight initialization
        init = RandomNormal(stddev=0.02)

        # Add a layer for taking the image as input
        input_image_layer = Input(shape=(784,))
        hidden_layer = Reshape((28, 28, 1), input_shape=(784,))(input_image_layer)

        # Downsample to 14x14
        hidden_layer = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)

        hidden_layer = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(hidden_layer)
        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)

        # Downsample to 7x7
        hidden_layer = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(hidden_layer)

        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)

        hidden_layer = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(hidden_layer)

        hidden_layer = LeakyReLU(alpha=0.2)(hidden_layer)
        hidden_layer = Dropout(0.5)(hidden_layer)

        # Flatten feature maps
        hidden_layer = Flatten()(hidden_layer)
        # Real/fake output
        realness_output = Dense(1, activation='sigmoid')(hidden_layer)
        # Class label output
        class_output = Dense(num_classes, activation='softmax')(hidden_layer)

        # Define model
        self.model = Model(input_image_layer, [realness_output, class_output])
        # Compile model
        self.opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.opt)

    def trainModelAndGetRealityAndClassLoss(self, images, fake_vs_real_labels, class_labels):
        """
        Get the real vs synthetic loss and the classification loss and train the model.

        Args:
            images:                 Image data.
            fake_vs_real_labels:    Labels for the images indicating if they were real or fake.
            class_labels:           Labels for the images indicating their class label.

        Returns: Loss for the real/synthetic classification and loss for the classification

        """
        _, real_sample_loss, class_loss = self.model.train_on_batch(images, [fake_vs_real_labels, class_labels])
        return (real_sample_loss, class_loss)

    def getClassification(self, images):
        """
        Get the output label for the given images.

        Args:
            images: Images to classify

        Returns:
            Output label for the images. Uses the index of maximum value in the 10-vector output that provides the
            score for each class

        """
        realness_score, class_labels_score = self.model.predict(images)
        class_labels = np.argmax(class_labels_score, axis=1)
        return class_labels

    def getClassificationAccuracy(self, test_images, test_labels):

        """
        Get the classification accuracy on the test data.

        Args:
            test_images: Images to classify.
            test_labels: Correct labels for the images.

        Returns: Classification accuracy
        """
        classification_results = self.getClassification(test_images)
        correct_label_vec = test_labels - classification_results

        # Count the number of incorrectly classified (non-zero) entries
        incorrect_classification_count = np.count_nonzero(correct_label_vec)
        accuracy_rate = (np.shape(test_labels)[0] - float(incorrect_classification_count)) / np.shape(test_labels)[0]

        return accuracy_rate

    def save(self, filename):
        """
        Save the model to a file
        Args:
            filename: Filename for saved model
        """
        self.model.save(filename)

class Generator(object):
    """
    Generator that can generate examples for a given class.
    """

    def __init__(self, latent_dim, num_classes):
        """
        Create the generator.

        Args:
            latent_dim:     Number of entries that represent a latent space input. (ex. 100 means the inputs will be
                            100-entry vectors.)
            num_classes:    Number of classes
        """

        # Weight initialization
        init = RandomNormal(stddev=0.02)

        # Class label input
        class_input_layer = Input(shape=(1,))

        # Embedding for categorical input
        class_input_hidden_layer = Embedding(num_classes, 50)(class_input_layer)
        # Linear multiplication
        n_nodes = 7 * 7
        class_input_hidden_layer = Dense(n_nodes, kernel_initializer=init)(class_input_hidden_layer)
        # Reshape to additional channel
        class_input_hidden_layer = Reshape((7, 7, 1))(class_input_hidden_layer)

        # Image generator input
        latent_space_input_layer = Input(shape=(latent_dim,))
        # Foundation for 7x7 image
        n_nodes = 384 * 7 * 7
        latent_space_hidden_layer = Dense(n_nodes, kernel_initializer=init)(latent_space_input_layer)
        latent_space_hidden_layer = Activation('relu')(latent_space_hidden_layer)
        latent_space_hidden_layer = Reshape((7, 7, 384))(latent_space_hidden_layer)

        # Merge image gen and label input
        merge_layer = Concatenate()([latent_space_hidden_layer, class_input_hidden_layer])

        # Upsample to 14x14
        merged_hidden_layer = Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merge_layer)
        merged_hidden_layer = Activation('relu')(merged_hidden_layer)
        # Upsample to 28x28
        merged_hidden_layer = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merged_hidden_layer)
        merged_hidden_layer = Activation('tanh')(merged_hidden_layer)
        output_image_layer = Flatten()(merged_hidden_layer)
        self.model = Model([latent_space_input_layer, class_input_layer], output_image_layer)

    def save(self, filename):
        """
        Save the generator to a file.
        Args:
            filename: Filename to save the generator to
        """
        self.model.save(filename)

    def generateData(self, latent_space_data, labels):
        """
        Generate data given the latent space representation and target class labels.

        Args:
            latent_space_data:  Latent space representation.
            labels:             Target class labels.

        Returns:
            Generated images for the target class labels constructed from the given latent space representations
        """
        return self.model.predict([latent_space_data, labels])

class Gan(object):
    """
    GAN that combines the discriminator and generator.
    """

    def __init__(self, generator, discriminator):
        """
        Create the GAN.

        Args:
            generator:      Generator (generates images).
            discriminator:  Discriminator (classifies images as real/fake and based on their class labels)
        """
        # Make the weights in the discriminator not trainable
        discriminator.model.trainable = False

        # Connect the generator and discriminator
        gan_output = discriminator.model(generator.model.output)

        # Define the inputs and outputs of the overall gan
        self.model = Model(generator.model.input, gan_output)

        # Compile the model
        self.opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.opt)

    def trainAndGetRealityAndClassLoss(self, latent_dim_representation, class_labels, realness_labels):
        """
        Train the model and get the losses for the generator portion of the real/fake discrimination objective and the
        classification objective.

        Args:
            latent_dim_representation:  Latent space representation.
            class_labels:               Class labels to generate images for.
            realness_labels:            Labels indicating if the images are real or fake.

        Returns:
            Loss for the generator for the fooling the discriminator objective and loss for the classifcation objective
        """

        _, generator_realness_training_loss, generator_class_training_loss = self.model.train_on_batch(
            [latent_dim_representation, class_labels], [realness_labels, class_labels])
        return generator_realness_training_loss, generator_class_training_loss

def selectRandomRealSamplesAndLabelAsReal(dataset, n_samples):
    """
    Select a random subset of images from the dataset and generate a label indicating that they are real.

    Args:
        dataset:    Tuples with full set of images and class labels, respectively
        n_samples:  Number of samples that should be in the subset.

    Returns:
        Tuple with the images and class labels for the images
    """

    images, class_labels = dataset
    # Choose indices of images/labels to keep randomly
    ix = randint(0, images.shape[0], n_samples)

    # Get the images and labels based on the random indices
    X, class_labels = tf.gather(images, ix), class_labels[ix]

    return (X, class_labels)

def createGeneratorInput(latent_dim, n_samples, n_classes):

    """
    Generate points in latent space as input for the generator.

    Args:
        latent_dim: Size of the latent space.
        n_samples:  Number of samples to generate.
        n_classes:  Number of possible classes.

    Returns:
        Tuple with the latent space samples and class labels
    """

    # Generate points in the latent space
    x_input = randn(latent_dim * n_samples)

    # Reshape into a batch of inputs for the network
    latent_inputs = x_input.reshape(n_samples, latent_dim)

    # Generate class labels randomly
    class_labels = randint(0, n_classes, n_samples)
    return (latent_inputs, class_labels)

def generateFakeSamples(generator, latent_dim, n_samples, num_classes):
    """
    Use the generator to generate n fake examples, with class labels

    Args:
        generator:      Generator that can generate fake images.
        latent_dim:     Latent space size.
        n_samples:      Number of samples to generate.
        num_classes:    Number of classes.
    Returns:
        Tuple of the generated images and the class labels for the images.
    """

    # generate points in latent space and random class labels
    z_input, labels_input = createGeneratorInput(latent_dim, n_samples, num_classes)

    # Generate images
    images = generator.generateData(z_input, labels_input)

    return images, labels_input


# generate samples and save as a plot and save the model
def summarizePerformanceAndPlotExamples(step, generator, discriminator, latent_dim, time_str, num_classes, n_samples=25):
    """
    Generate samples and save as a plot and save the model.
    
    Args:
        step:           Step number
        generator:        Generator
        discriminator:        Discriminator
        latent_dim:     Latent space size
        time_str:       String representing the program execution time.
        num_classes:    Number of classes.
        n_samples:      Number of samples to generate
    """

    # Come up with some fake examples
    fake_images, labels = generateFakeSamples(generator, latent_dim, n_samples, num_classes)

    # Scale from [-1,1] to [0,1]
    fake_images = (fake_images + 1) / 2.0

    # Plot images
    for i in range(n_samples):

        # Define subplot
        ax = pyplot.subplot(5, 5, 1 + i)
        pyplot.axis('off')
        plot_X = tf.reshape(fake_images, [-1, 28, 28, 1])
        ax.set_title("label: " + str(labels[i]), fontsize=10)
        pyplot.imshow(plot_X[i, :, :, 0])

    # Save plot to file
    plot_filename = ("generated_plot_" + time_str + "_%04d.png") % (step + 1)

    plt.subplots_adjust(hspace=0.5)
    pyplot.savefig(plot_filename)
    pyplot.close()

    # Save the generator and discriminator
    generator_filename = ("gen_model_" + time_str + "_%04d.h5") % (step + 1)
    generator.save(generator_filename)

    discriminator_filename = ("disc_model_" + time_str + "_%04d_.h5") % (step + 1)
    discriminator.save(discriminator_filename)
    print('>Saved: %s and %s and %s' % (plot_filename, generator_filename, discriminator_filename))

class Losses(object):
    """
    Losses for both models at a point in time.
    """

    def __init__(self, generator_train_realness_loss, generator_class_loss, discriminator_realness_loss,
                 discriminator_class_loss):
        self.generator_train_realness_loss = generator_train_realness_loss
        self.generator_class_loss = generator_class_loss
        self.discriminator_class_loss = discriminator_class_loss
        self.discriminator_fool_loss = discriminator_realness_loss


    def __str__(self):
        return "Losses(discrim_real:" + str(self.discriminator_fool_loss) + ",discrim_class:" + str(self.discriminator_class_loss) + ",gen_fool:" + str(self.generator_train_realness_loss) \
        + ",gen_class:" + str(self.generator_class_loss) + ")"




def runTrainingIterationOnBatch(real_images, real_image_labels, discriminator, generator, full_gan,
                                test_latent_data, test_latent_classes, latent_dim, num_classes):

    """
    Run a training iteration on one batch of images.

    Args:
        real_images:            Real images to train the discriminator on.
        real_image_labels:      Class labels for the real images.
        discriminator:          Discriminator to train.
        generator:              Generator to use for generating fake samples.
        full_gan:               Full GAN. Used to implicitly train the generator.
        test_latent_data:       Latent data that we should generate images for at each iteration
        test_latent_classes:    Classes that we should generate images for at each iteration
        latent_dim:             Latent space size.
        num_classes:            Number of classes

    Returns:
        Images for the test latent data and classes, and the losses for the training iteration
    """

    num_samples = real_images.shape[0]
    realness_label_real_samples = np.ones((num_samples, 1))

    # Train the discriminator using real samples
    (discriminator_realness_training_loss_real_samples, discriminator_class_training_loss_real_samples) = \
        discriminator.trainModelAndGetRealityAndClassLoss(real_images, realness_label_real_samples, real_image_labels)

    # Create input to the generator
    latent_dim_reps, fake_image_class_labels = createGeneratorInput(latent_dim, num_samples * 2, num_classes)

    # Generate fake images for training the discriminator
    fake_images = generator.generateData(latent_dim_reps[:num_samples, :], fake_image_class_labels[:num_samples])
    fake_image_realness_for_discriminator = zeros((num_samples, 1))

    # Train the discriminator on the fake images
    (discriminator_realness_training_loss_fake_samples, discriminator_class_training_loss_fake_samples) = \
        discriminator.trainModelAndGetRealityAndClassLoss(fake_images, fake_image_realness_for_discriminator, fake_image_class_labels[:num_samples])

    # Train the generator via the GAN on the latent input
    fake_image_realness_for_generator = ones((2 * num_samples, 1))
    generator_realness_training_loss, generator_class_training_loss = \
        full_gan.trainAndGetRealityAndClassLoss(latent_dim_reps, fake_image_class_labels,
                                                fake_image_realness_for_generator)

    losses = Losses(generator_class_loss=generator_class_training_loss,
                    generator_train_realness_loss=generator_realness_training_loss,
                    discriminator_class_loss=(discriminator_class_training_loss_real_samples + discriminator_class_training_loss_fake_samples),
                    discriminator_realness_loss=(discriminator_realness_training_loss_fake_samples + discriminator_realness_training_loss_real_samples))

    # Generate images for the test latent data and classes
    generated_images = generator.generateData(test_latent_data, test_latent_classes)

    return (generated_images, losses)


def trainGanByBatches(generator, discriminator, full_gan, dataset, latent_dim, test_latent_data, test_gen_classes,
                      n_epochs=100, n_batch=64):
    """
    Train the GAN in batches for the given number of epochs.

    Args:
        generator:          Generator to use for generating fake samples.
        discriminator:      Discriminator to train.
        full_gan:           Full GAN. Used to implicitly train the generator.
        dataset:            Full dataset to train on.
        latent_dim:         Latent space size.
        test_latent_data:   Latent data that we should generate images for at each iteration
        test_gen_classes:   Classes that we should generate images for at each iteration
        n_epochs:           Number of epochs to train for.
        n_batch:            Number of images per batch.

    Returns:
        Losses by training iteration and images for the test latent data and classes by epoch
    """

    date_start_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    checkpoint_dir = './training_checkpoints_' + date_start_str
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=full_gan.opt,
                                     discriminator_optimizer=discriminator.opt,
                                     generator=generator.model,
                                     discriminator=discriminator.model)

    # Calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print("Bat per epo " + str(bat_per_epo))
    print("Epochs " + str(n_epochs))

    # Calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print("Number of steps: " + str(n_steps))

    # Calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)

    loss_by_iter = {}
    generated_images_by_epoch = {}

    num_classes = 10

    for i in range(n_steps):

        # Generate real images and labels to train on
        real_images, labels_real = selectRandomRealSamplesAndLabelAsReal(dataset, half_batch)

        # Run the training iteration
        generated_test_images, losses = runTrainingIterationOnBatch(real_images, labels_real, discriminator, generator,
                                                                    full_gan, test_latent_data, test_gen_classes,
                                                                    latent_dim, num_classes)

        # Save the losses
        loss_by_iter[i] = losses

        # If at the end of an epoch, save the imaages for the test latent dimensions
        if ((i + 1) % (bat_per_epo)) == 0:
            generated_images_by_epoch[math.trunc(i / bat_per_epo)] = generated_test_images

        if ((i % 10) == 0):
            print("Losses " + str(i) + ": "  + str(losses))

        if ((i % 100) == 0):
            print("Iteration " + str(i))

        if ((i + 1) % (bat_per_epo * 10)) == 0:

            summarizePerformanceAndPlotExamples(i, generator, discriminator, latent_dim, date_start_str, num_classes)

            gan_new_fpath = "gan_results_iter_" + str(i) + "_" + date_start_str + ".pkl"
            train_size = dataset[0].shape[0]
            joblib.dump((train_size, loss_by_iter, generated_images_by_epoch, n_batch, n_epochs), gan_new_fpath)
            checkpoint.save(file_prefix=checkpoint_prefix)

    summarizePerformanceAndPlotExamples(n_steps, generator, discriminator, latent_dim, date_start_str, num_classes)
    return loss_by_iter, generated_images_by_epoch

