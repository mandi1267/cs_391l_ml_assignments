import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

class CnnClassifier(object):

    def __init__(self):
        pass

    def trainModel(self, training_data, training_labels):
        self.model = models.Sequential()

        self.model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        # self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

        self.model.summary()

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = self.model.fit(training_data, training_labels, epochs=5)
        #                     ,validation_data=(test_images, test_labels))

        # plt.plot(history.history['accuracy'], label='accuracy')
        # # plt.plot(history.history['val_accuracy'], label='val_accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.ylim([0.5, 1])
        # plt.legend(loc='lower right')
        # plt.show()


    def testModel(self, test_data, test_labels):
        test_loss, test_acc = self.model.evaluate(test_data,  test_labels, verbose=2)
        print(test_acc)
        return test_acc


