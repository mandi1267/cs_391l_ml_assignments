from tensorflow.keras import layers
import tensorflow as tf

class GanDiscriminator(object):

    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
        self.model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def getRealOrFakeClassification(self, flattened_images, training=False):
        return self.model(flattened_images, training=training)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def optimize(self, gradients_of_discriminator, ):
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.model.trainable_variables))



