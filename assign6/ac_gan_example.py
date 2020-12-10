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


# define the standalone discriminator model
class Discriminator(object):
    def __init__(self):
        self.n_classes = 10
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
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # downsample to 7x7
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # normal
        fe = Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # real/fake output
        out1 = Dense(1, activation='sigmoid')(fe)
        # class label output
        out2 = Dense(self.n_classes, activation='softmax')(fe)
        # define model
        self.model = Model(in_image, [out1, out2])
        # compile model
        self.opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.opt)

    def getRealityAndClassLoss(self, images, fake_vs_real_labels, class_labels):
        _, real_sample_loss, class_loss = self.model.train_on_batch(images, [fake_vs_real_labels, class_labels])
        return (real_sample_loss, class_loss)

    def save(self, filename):
        self.model.save(filename)

def plotImages(predictions, iter, labels):
    # plt.title("Images, iter " + str(iter) + str(labels[:16]))

    for i in range(min(16, predictions.shape[0])):
        ax = plt.subplot(4, 4, i + 1)
        # print(labels[i])
        reshaped_pred = predictions[i]
        reshaped_pred = tf.reshape(reshaped_pred, [28, 28])
        plt.imshow(reshaped_pred * 127.5 + 127.5)
        plt.axis('off')
        ax.set_title("label: " + str(labels[i]))
    plt.suptitle("Images, iter " + str(iter))
    plt.show()


# define the standalone generator model
def define_generator(latent_dim, n_classes=10):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 50)(in_label)
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
    gen = BatchNormalization()(gen)
    gen = Activation('relu')(gen)
    # upsample to 28x28
    gen = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('tanh')(gen)
    out_layer = Flatten()(gen)
    # define model
    model = Model([in_lat, in_label], out_layer)
    return model

class Gan(object):

    def __init__(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.model.trainable = False
        # connect the outputs of the generator to the inputs of the discriminator
        gan_output = discriminator.model(generator.output)
        # define gan model as taking noise and label and outputting real/fake and label outputs
        self.model = Model(generator.input, gan_output)
        # compile model
        self.opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=self.opt)

# # define the combined generator and discriminator model, for updating the generator
# def define_gan(g_model, d_model):



# load images
def load_real_samples():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # load dataset
    # (trainX, trainy), (_, _) = load_data()
    # expand to 3d, e.g. add channels
    X = expand_dims(train_images, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    X = tf.reshape(X, [-1, 784, 1])
    print(X.shape, train_labels.shape)
    return [X, train_labels]


# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = tf.gather(images, ix), labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, d_model, latent_dim, time_str="", n_samples=25):
    # prepare fake examples
    [X, labels], _ = generate_fake_samples(g_model, latent_dim, n_samples)
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
    filename1 = ("generated_plot_%04d_" + time_str + ".png") % (step + 1)

    plt.subplots_adjust(hspace=0.5)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = ('gen_model_%04d_' + time_str + '.h5') % (step + 1)
    g_model.save(filename2)
    d_model_filename = ('disc_model_%04d_' + time_str + '.h5') % (step + 1)
    d_model.save(d_model_filename)
    print('>Saved: %s and %s' % (filename1, filename2))


class Losses(object):

    def __init__(self, generator_train_fool_loss, generator_class_loss, discriminator_fool_loss,
                 discriminator_class_loss):
        self.generator_train_fool_loss = generator_train_fool_loss
        self.generator_class_loss = generator_class_loss
        self.discriminator_class_loss = discriminator_class_loss
        self.discriminator_fool_loss = discriminator_fool_loss


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
    date_start_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    checkpoint_dir = './training_checkpoints_' + date_start_str
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gan_model.opt,
                                     discriminator_optimizer=d_model.opt,
                                     generator=generator,
                                     discriminator=d_model.model)

    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    print("Bat per epo " + str(bat_per_epo))
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print("Number of steps: " + str(n_steps))
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)

    loss_by_iter = {}

    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        (real_img_loss_real_imgs, class_loss_real_imgs) = d_model.getRealityAndClassLoss(X_real, y_real, labels_real)
        # _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])

        # generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # print(X_fake.shape)

        # update discriminator model weights
        (real_img_loss_fake_imgs, class_loss_fake_imgs) = d_model.getRealityAndClassLoss(X_fake, y_fake, labels_fake)
        # _, d_f, d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        _, g_1, g_2 = gan_model.model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i + 1, real_img_loss_real_imgs, class_loss_real_imgs, real_img_loss_fake_imgs, class_loss_fake_imgs, g_1, g_2))

        loss_by_iter[i] = Losses(generator_train_fool_loss=g_1, generator_class_loss=g_2,
                                 discriminator_fool_loss=(real_img_loss_real_imgs + real_img_loss_fake_imgs), discriminator_class_loss=(class_loss_real_imgs + class_loss_fake_imgs))
        # evaluate the model performance every 'epoch'
        if (i + 1) % (bat_per_epo) == 0:
            # plotImages(X_fake, i, labels_fake)
            summarize_performance(i, g_model, d_model, latent_dim, date_start_str)

            gan_new_fpath = "gan_results_iter_" + str(i) + "_" + date_start_str + ".pkl"
            train_size = dataset[0].shape[0]
            joblib.dump((train_size, loss_by_iter, n_batch, n_epochs), gan_new_fpath)
        if ((i + 1) % (bat_per_epo * 10)) == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    summarize_performance(n_steps, g_model, d_model, latent_dim, date_start_str)
    return loss_by_iter


if __name__ == '__main__':
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    discrim_obj = Discriminator()
    # discriminator, discrim_opt = define_discriminator()
    # create the generator
    generator = define_generator(latent_dim)
    # create the gan
    # gan_model, gen_opt = define_gan(generator, discriminator)
    gan_model = Gan(generator, discrim_obj)
    # load image data
    dataset = load_real_samples()
    # train model
    training_subset_size = 10000
    n_epochs = 500
    n_batch = 64
    subset = generate_real_samples(dataset, training_subset_size)[0]
    loss_by_iter = train(generator, discrim_obj, gan_model, subset, latent_dim,
                         n_epochs=n_epochs, n_batch=n_batch)
