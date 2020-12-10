import joblib
# from gan_example import *
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import collections

class Losses(object):

    def __init__(self, generator_train_fool_loss, generator_class_loss, discriminator_fool_loss, discriminator_class_loss):
        self.generator_train_fool_loss = generator_train_fool_loss
        self.generator_class_loss = generator_class_loss
        self.discriminator_class_loss = discriminator_class_loss
        self.discriminator_fool_loss = discriminator_fool_loss


def plotLossesByIter(loss_by_iter):
    #
    # epochs = set([iter for (epoch, iter) in loss_by_iter.keys()])
    # epochs = [epoch for epoch in epochs]
    #
    # discrim_loss_by_epoch = collections.defaultdict(lambda: 0)
    # gen_loss_by_epoch = collections.defaultdict(lambda: 0)
    #
    #
    # for ((epoch, iter), (gen_loss, disc_loss)) in loss_by_iter.items():
    #     gen_loss_by_epoch[iter] += gen_loss
    #     discrim_loss_by_epoch[iter] += disc_loss
    #
    # print(epochs)
    #
    #
    # generator_fool_loss = [gen_loss_by_epoch[epoch] for epoch in epochs]
    # discriminator_fool_loss = [discrim_loss_by_epoch[epoch] for epoch in epochs]

    iterations_nums = [iter for iter in loss_by_iter.keys()]

    generator_fool_loss = [loss_by_iter[iteration].generator_train_fool_loss for iteration in iterations_nums]
    discriminator_fool_loss = [loss_by_iter[iteration].discriminator_fool_loss for iteration in iterations_nums]

    plt.plot(iterations_nums, generator_fool_loss, label="Generator loss")
    plt.plot(iterations_nums, discriminator_fool_loss, label="Discriminator loss")


    # plt.plot(epochs, generator_fool_loss, label="Generator loss")
    # plt.plot(epochs, discriminator_fool_loss, label="Discriminator loss")

    plt.xlabel("Iteration number")
    plt.ylabel("Loss")
    plt.title("Discriminator and generator losses")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    results_file_name = "gan_results_iter_15599_2020-12-08T21:11:18.pkl"

    (train_size, loss_by_iter, n_batch, n_epochs) = joblib.load(results_file_name)
    #
    # results_file_name = "tf_gan_results_iter_900_2020-12-09T20:39:05.pkl"
    # results_file_name = "tf_gan_results_iter_1000_2020-12-09T20:51:11.pkl"
    # (losses, n_epochs) = joblib.load(results_file_name)
    # print(losses.keys())

    # print(train_size)
    # print(len(loss_by_iter))
    # print(n_batch)
    # print(n_epochs)
    #
    # plotLossesByIter(losses)
    plotLossesByIter(loss_by_iter)

    # generator_optimizer = None
    # discriminator_optimizer = None
    # generator = None
    # discriminator = None
    #
    # checkpoint_dir = './training_checkpoints_2020-12-08T21:11:18'
    # # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    # #                                  discriminator_optimizer=discriminator_optimizer,
    # #                                  generator=generator,
    # #                                  discriminator=discriminator)
    #
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
