import tensorflow as tf

# import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time
import datetime
from assign6 import  *

import joblib

from mnist_reader import *

def getMNistData(mnist_data_file, mnist_label_file):
    img_data, num_rows_in_img, num_cols_in_img = readMNISTData(mnist_data_file)
    labels = readMNISTLabels(mnist_label_file)
    return img_data, labels

def normalizeImages(img_data):
    return (img_data - 127.5) / 127.5  # Normalize the images to [-1, 1]


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    model.add(layers.Flatten())

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Reshape((28, 28, 1), input_shape=[784,]))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(cross_entropy_func, real_output, fake_output):
    real_loss = cross_entropy_func(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_func(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(cross_entropy_func, fake_output):
    return cross_entropy_func(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, batch_size, noise_dimensions, gen_model, disc_model, cross_entropy_func):
    noise = tf.random.normal([batch_size, noise_dimensions])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = gen_model(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(cross_entropy_func, fake_output)
      disc_loss = discriminator_loss(cross_entropy_func, real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc_model.trainable_variables))

    return (gen_loss, disc_loss)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      unflattened = tf.reshape(predictions[i, :], [-1, 28, 28])
      print(unflattened.shape)
      plt.imshow(unflattened[0, :] * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('image_at_epoch_20000_{:04d}.png'.format(epoch))
  plt.show(block=False)

def train(dataset, epochs, batch_size, noise_dimensions, gen_model, disc_model, seed, checkpt, checkpt_prefix, cross_entropy_func):
  total_iter = 0
  date_start_str = datetime.datetime.now().replace(microsecond=0).isoformat()

  losses = {}
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      step_losses = train_step(image_batch, batch_size, noise_dimensions, gen_model, disc_model, cross_entropy_func)
      losses[(epoch, total_iter)] = step_losses
      total_iter += 1

    # Produce images for the GIF as we go
    # display.clear_output(wait=True)
    generate_and_save_images(gen_model,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpt.save(file_prefix = checkpt_prefix)
      gan_new_fpath = "tf_gan_results_iter_" + str(total_iter) + "_" + date_start_str + ".pkl"
      joblib.dump((losses, epochs), gan_new_fpath)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  # display.clear_output(wait=True)
  generate_and_save_images(gen_model,
                           epochs,
                           seed)
  gan_new_fpath = "tf_gan_results_iter_" + str(total_iter) + "_" + date_start_str + ".pkl"
  joblib.dump((losses, epochs), gan_new_fpath)

if __name__ == '__main__':
    training_data_file_name = "/Users/mandiadkins/Downloads/train-images.idx3-ubyte"
    training_label_file_name = "/Users/mandiadkins/Downloads/train-labels.idx1-ubyte"

    test_data_file_name = "/Users/mandiadkins/Downloads/t10k-images.idx3-ubyte"
    test_label_file_name = "/Users/mandiadkins/Downloads/t10k-labels.idx1-ubyte"

    train_images, train_labels = getMNistData(training_data_file_name, training_label_file_name)
    train_images, train_labels = getRandomSubsetOfData(train_images, train_labels, 20000)

    train_images = normalizeImages(train_images)

    BUFFER_SIZE = 5000
    BATCH_SIZE = 256

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    generated_image_not_flattened = tf.reshape(generated_image[0, :], [-1, 28, 28])

    plt.imshow(generated_image_not_flattened[0, :, :])

    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print (decision)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    date_start_str = datetime.datetime.now().replace(microsecond=0).isoformat()
    checkpoint_dir = './training_checkpoints_' + date_start_str
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    train(train_dataset, EPOCHS, BATCH_SIZE, noise_dim, generator, discriminator, seed, checkpoint, checkpoint_prefix,
          cross_entropy)

