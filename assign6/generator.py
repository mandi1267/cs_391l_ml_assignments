from tensorflow.keras import layers
import tensorflow as tf

class Generator(object):

    def __init__(self):

        # TODO this is just copied -- what modifications do we need to make
        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Reshape((7, 7, 256)))
        assert self.model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 7, 7, 128)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert self.model.output_shape == (None, 14, 14, 64)
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

        self.model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.model.output_shape == (None, 28, 28, 1)
        self.model.add(layers.Flatten())

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)



    def generateData(self, input_data, training=False):
        # noise = tf.random.normal([1, 100])
        generated_images = self.model(input_data, training=training)

        return generated_images

        # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        # pass

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def optimize(self, gradients_of_generator):
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.model.trainable_variables))

