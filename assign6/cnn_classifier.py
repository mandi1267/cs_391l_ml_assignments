import tensorflow as tf

from tensorflow.keras import layers, models

class ClassifierWrapper(object):

    def __init__(self, underlying_model):
        self.underlying_model = underlying_model

    def trainModel(self, training_data, training_labels, num_epochs):
        self.underlying_model.summary()

        self.underlying_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = self.underlying_model.fit(training_data, training_labels, epochs=num_epochs)

    def testModel(self, test_data, test_labels):
        test_loss, test_acc = self.underlying_model.evaluate(test_data,  test_labels, verbose=2)
        print(test_acc)
        return test_acc

class CnnClassifier(ClassifierWrapper):

    def __init__(self):
        super(CnnClassifier, self).__init__(self.createCnnModel())

    def createCnnModel(self):
        cnn_model = models.Sequential()

        cnn_model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
        cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        cnn_model.add(layers.MaxPooling2D((2, 2)))
        cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        cnn_model.add(layers.Flatten())
        cnn_model.add(layers.Dense(64, activation='relu'))
        cnn_model.add(layers.Dense(10))

        cnn_model.summary()
        return cnn_model

