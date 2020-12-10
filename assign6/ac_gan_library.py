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
from tensorflow.keras.layers import BatchNormalization
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
    Normalize the image data to be in the range -1 to 1 and make sure it is floats.

    TODO does this work on numpy or only tensorflow types?
    :param img_data:
    :return:
    """
    img_data = img_data.astype('float32')
    # scale from [0,255] to [-1,1]
    return (img_data - 127.5) / 127.5

def normalizeToIntRange(img_data):
    """
    Normalize the image data to be in the range 0 to 255
    :param img_data:
    :return:
    """
    return img_data * 127.5 + 127.5


# define the standalone discriminator model
class Discriminator(object):
    def __init__(self, num_classes):
        # weight initialization
        init = RandomNormal(stddev=0.02)

        # image input
        in_image = Input(shape=(784,))
        fe = Reshape((28, 28, 1), input_shape=(784,))(in_image)
        # downsample to 14x14
        fe = Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # normal
        fe = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(fe)
        # fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # downsample to 7x7
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
        # fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # normal
        fe = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(fe)
        # fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # real/fake output
        out1 = Dense(1, activation='sigmoid')(fe)
        # class label output
        out2 = Dense(num_classes, activation='softmax')(fe)
        # define model
        self.model = Model(in_image, [out1, out2])
        # compile model
        self.opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.opt)

    def getRealityAndClassLoss(self, images, fake_vs_real_labels, class_labels):
        _, real_sample_loss, class_loss = self.model.train_on_batch(images, [fake_vs_real_labels, class_labels])
        return (real_sample_loss, class_loss)

    def getClassification(self, images):
        realness_score, class_labels_score = self.model.predict(images)
        class_labels = np.argmax(class_labels_score, axis=1)
        return class_labels

    def getClassificationAccuracy(self, test_images, test_labels):
        classification_results = self.getClassification(test_images)
        correct_label_vec = test_labels - classification_results

        # Count the number of incorrectly classified (non-zero) entries
        incorrect_classification_count = np.count_nonzero(correct_label_vec)
        accuracy_rate = (np.shape(test_labels)[0] - float(incorrect_classification_count)) / np.shape(test_labels)[0]

        return accuracy_rate

    def save(self, filename):
        self.model.save(filename)

def plotImages(predictions, iter, labels):
    # plt.title("Images, iter " + str(iter) + str(labels[:16]))

    for i in range(min(16, predictions.shape[0])):
        ax = plt.subplot(4, 4, i + 1)
        reshaped_pred = predictions[i]
        reshaped_pred = tf.reshape(reshaped_pred, [28, 28])
        plt.imshow(normalizeToIntRange(reshaped_pred))
        plt.axis('off')
        ax.set_title("label: " + str(labels[i]))
    plt.suptitle("Images, iter " + str(iter))
    plt.show()


# define the standalone generator model
class Generator(object):
    def __init__(self, latent_dim, num_classes):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(num_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = Dense(n_nodes, kernel_initializer=init)(li)
        # reshape to additional channel
        li = Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        # foundation for 7x7 image
        n_nodes = 384 * 7 * 7
        gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
        gen = Activation('relu')(gen)
        gen = Reshape((7, 7, 384))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        # upsample to 14x14
        gen = Conv2DTranspose(192, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(merge)
        # gen = BatchNormalization()(gen)
        gen = Activation('relu')(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
        gen = Activation('tanh')(gen)
        out_layer = Flatten()(gen)
        # define model
        self.model = Model([in_lat, in_label], out_layer)

    def save(self, filename):
        self.model.save(filename)

    def generateData(self, latent_space_data, labels):
        return self.model.predict([latent_space_data, labels])

class Gan(object):

    def __init__(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.model.trainable = False
        # connect the outputs of the generator to the inputs of the discriminator
        gan_output = discriminator.model(generator.model.output)
        # define gan model as taking noise and label and outputting real/fake and label outputs
        self.model = Model(generator.model.input, gan_output)
        # compile model
        self.opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.opt)

    def trainAndGetRealityAndClassLoss(self, latent_dim_representation, class_labels, realness_labels):
        _, generator_realness_training_loss, generator_class_training_loss = self.model.train_on_batch(
            [latent_dim_representation, class_labels], [realness_labels, class_labels])
        return generator_realness_training_loss, generator_class_training_loss

# load images
def load_real_samples():
    # TODO read from MNIST files rather than tf datasets
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # load dataset
    # (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(train_images, axis=-1)
    # convert from ints to floats
    # X = X.astype('float32')
    # # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    #
    X = normalizeToFloatRange(X)
    X = tf.reshape(X, [-1, 784, 1])
    print(X.shape, train_labels.shape)
    return [X, train_labels]


# select real samples
def selectRandomRealSamplesAndLabelAsReal(dataset, n_samples):
    # split into images and labels
    images, class_labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, class_labels = tf.gather(images, ix), class_labels[ix]
    # Generate label indicating that the samples are real
    realness_labels = ones((n_samples, 1))
    return (X, class_labels, realness_labels)


# generate points in latent space as input for the generator
def createGeneratorInput(latent_dim, n_samples, n_classes):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return (z_input, labels)


# use the generator to generate n fake examples, with class labels
def generateFakeSamples(generator, latent_dim, n_samples, num_classes):
    # generate points in latent space
    z_input, labels_input = createGeneratorInput(latent_dim, n_samples, num_classes)
    # predict outputs
    images = generator.generateData(z_input, labels_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, labels_input, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, latent_dim, time_str, num_classes, n_samples=25):
    # prepare fake examples
    X, labels, _ = generateFakeSamples(g_model, latent_dim, n_samples, num_classes)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(n_samples):
        # define subplot
        ax = pyplot.subplot(5, 5, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        plot_X = tf.reshape(X, [-1, 28, 28, 1])
        ax.set_title("label: " + str(labels[i]), fontsize=10)
        pyplot.imshow(plot_X[i, :, :, 0])
    # save plot to file
    plot_filename = ("generated_plot_" + time_str + "_%04d.png") % (step + 1)

    plt.subplots_adjust(hspace=0.5)
    pyplot.savefig(plot_filename)
    pyplot.close()
    # save the generator model
    generator_filename = ("gen_model_" + time_str + "_%04d.h5") % (step + 1)
    g_model.save(generator_filename)
    d_model_filename = ("disc_model_" + time_str + "_%04d_.h5") % (step + 1)
    d_model.save(d_model_filename)
    print('>Saved: %s and %s and %s' % (plot_filename, generator_filename, d_model_filename))

class Losses(object):

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
    # get randomly selected 'real' samples

    num_samples = real_images.shape[0]
    # print("Num samples " + str(num_samples))
    realness_label_real_samples = np.ones((num_samples, 1))

    # Train the discriminator using real samples
    (discriminator_realness_training_loss_real_samples, discriminator_class_training_loss_real_samples) = \
        discriminator.getRealityAndClassLoss(real_images, realness_label_real_samples, real_image_labels)

    latent_dim_reps, fake_image_class_labels = createGeneratorInput(latent_dim, num_samples * 2, num_classes)
    fake_images = generator.generateData(latent_dim_reps[:num_samples, :], fake_image_class_labels[:num_samples])

    fake_image_realness_for_generator = ones((2 * num_samples, 1))
    fake_image_realness_for_discriminator = zeros((num_samples, 1))

    # update discriminator model weights
    (discriminator_realness_training_loss_fake_samples, discriminator_class_training_loss_fake_samples) = \
        discriminator.getRealityAndClassLoss(fake_images, fake_image_realness_for_discriminator, fake_image_class_labels[:num_samples])

    # update the generator via the discriminator's error
    generator_realness_training_loss, generator_class_training_loss = \
        full_gan.trainAndGetRealityAndClassLoss(latent_dim_reps, fake_image_class_labels,
                                                fake_image_realness_for_generator)

    generator_realness_training_loss, generator_class_training_loss = \
        full_gan.trainAndGetRealityAndClassLoss(latent_dim_reps, fake_image_class_labels,
                                                fake_image_realness_for_generator)

    losses = Losses(generator_class_loss=generator_class_training_loss,
                    generator_train_realness_loss=generator_realness_training_loss,
                    discriminator_class_loss=(discriminator_class_training_loss_real_samples + discriminator_class_training_loss_fake_samples),
                    discriminator_realness_loss=(discriminator_realness_training_loss_fake_samples + discriminator_realness_training_loss_real_samples))

    generated_images = generator.generateData(test_latent_data, test_latent_classes)

    return (generated_images, losses)


# train the generator and discriminator
def trainGanByBatches(g_model, d_model, gan_model, dataset, latent_dim, test_latent_data, test_gen_classes, n_epochs=100, n_batch=64):
    date_start_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    checkpoint_dir = './training_checkpoints_' + date_start_str
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gan_model.opt,
                                     discriminator_optimizer=d_model.opt,
                                     generator=g_model.model,
                                     discriminator=d_model.model)

    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print("Bat per epo " + str(bat_per_epo))
    print("Epochs " + str(n_epochs))
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print("Number of steps: " + str(n_steps))
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)

    loss_by_iter = {}
    generated_images_by_iter = {}

    num_classes = 10

    # manually enumerate epochs
    for i in range(n_steps):

        X_real, labels_real, y_real = selectRandomRealSamplesAndLabelAsReal(dataset, half_batch)

        generated_test_images, losses = runTrainingIterationOnBatch(X_real, labels_real, d_model, g_model, gan_model,
                                test_latent_data, test_gen_classes, latent_dim, num_classes)
        loss_by_iter[i] = losses
        generated_images_by_iter[i] = generated_test_images

        if ((i % 10) == 0):
            print("Losses " + str(i) + ": "  + str(losses))

        if ((i % 100) == 0):
            print("Iteration " + str(i))

        if ((i + 1) % (bat_per_epo * 10)) == 0:

            summarize_performance(i, g_model, d_model, latent_dim, date_start_str, num_classes)

            gan_new_fpath = "gan_results_iter_" + str(i) + "_" + date_start_str + ".pkl"
            train_size = dataset[0].shape[0]
            joblib.dump((train_size, loss_by_iter, generated_images_by_iter, n_batch, n_epochs), gan_new_fpath)
            checkpoint.save(file_prefix=checkpoint_prefix)

    summarize_performance(n_steps, g_model, d_model, latent_dim, date_start_str, num_classes)
    return loss_by_iter, generated_images_by_iter


# if __name__ == '__main__':
#
#     # size of the latent space
#     latent_dim = 100
#     num_classes = 10
#     # create the discriminator
#     discrim_obj = Discriminator(num_classes)
#     # discriminator, discrim_opt = define_discriminator()
#     # create the generator
#     generator = Generator(latent_dim, num_classes)
#     # generator = define_generator(latent_dim)
#     # create the gan
#     # gan_model, gen_opt = define_gan(generator, discriminator)
#     gan_model = Gan(generator, discrim_obj)
#     # load image data
#     dataset = load_real_samples()
#     # train model
#     training_subset_size = 10000
#     n_epochs = 500
#     n_batch = 64
#
#     test_latent_representation, test_labels = createGeneratorInput(latent_dim, 25, num_classes)
#
#     subset_imgs, subset_labels, subset_realness_labels = selectRandomRealSamplesAndLabelAsReal(dataset, training_subset_size)
#     loss_by_iter = train(generator, discrim_obj, gan_model, (subset_imgs, subset_labels), latent_dim,
#                          test_latent_representation, test_labels, n_epochs=n_epochs, n_batch=n_batch)
