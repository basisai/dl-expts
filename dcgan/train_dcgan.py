"""
Script to train DCGAN.
"""
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten,  Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Conv2DTranspose, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from keras.utils import to_categorical


plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

BATCH_SIZE = 128
EPOCHS = 3
NOISE_DIM = 96


def save_images(images, it, img_rows=28, img_cols=28):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.imshow(img.reshape(img_rows, img_cols))

    fig.savefig("/artefact/plots/image_iter_{}.png".format(it))
    plt.close(fig)


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def load_mnist_data(dim):
    img_rows = 28
    img_cols = 28
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(len(x_train), "train samples")
    print(len(x_test), "test samples")

    if dim == 3:
        if K.image_data_format() == "channels_first":
            x_train = x_train.reshape(len(x_train), 1, img_rows, img_cols)
            x_test = x_test.reshape(len(x_test), 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(len(x_train), img_rows, img_cols, 1)
            x_test = x_test.reshape(len(x_test), img_rows, img_cols, 1)
    elif dim == 1:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)

    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


class DatasetBatches(object):
    def __init__(self, x_train, shuffle=True):
        self.x_train = preprocess_img(x_train)
        self.shuffle = shuffle

    def batches(self, BATCH_SIZE):
        n = len(self.x_train)
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.x_train = self.x_train[idx]

        n_batches = n // BATCH_SIZE
        for i in range(0, n_batches * BATCH_SIZE, BATCH_SIZE):
            yield self.x_train[i:i + BATCH_SIZE]


def build_discriminator():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=5, padding="valid")(inputs)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
    x = Conv2D(64, kernel_size=5, padding="valid")(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.01)(x)
    probs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, probs)
    return model


def build_generator(noise_dim):
    inputs = Input(shape=(noise_dim,))
    x = Dense(1024, activation="relu")(inputs)
    x = BatchNormalization()(x, training=True)
    x = Dense(7 * 7 * 128, activation="relu")(x)
    x = BatchNormalization()(x, training=True)
    x = Reshape((7, 7, 128))(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, activation="relu", padding="same")(x)
    x = BatchNormalization()(x, training=True)
    img = Conv2DTranspose(1, kernel_size=4, strides=2, activation="tanh", padding="same")(x)

    model = Model(inputs, img)
    return model


def gan_loss(discriminator, combined, imgs, gen_imgs, z):
    """Compute the GAN loss.

    Returns:
    - d_loss: discriminator loss scalar
    - g_loss: generator loss scalar
    """
    # Adversarial ground truths
    valid = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    # Train Discriminator (real classified as ones and generated as zeros)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator (wants discriminator to mistake images as real)
    g_loss = combined.train_on_batch(z, valid)
    return d_loss, g_loss


def train(generator, discriminator, combined, x_train,
          batch_size=128, epochs=10, noise_dim=96, show_every=250, print_every=50):
    it = 0
    for epoch in range(epochs):
        for imgs in DatasetBatches(x_train).batches(batch_size):

            # Sample noise and generate a batch of new images
            z = np.random.uniform(-1, 1, (batch_size, noise_dim))
            gen_imgs = generator.predict(z)

            # every show often, show a sample result
            if it % show_every == 0:
                save_images(gen_imgs[:16], it)

            # Train Discriminator & Generator
            d_loss, g_loss = gan_loss(discriminator, combined, imgs, gen_imgs, z)

            # print loss every so often.
            # We want to make sure D_loss doesn"t go to 0
            if it % print_every == 0:
                print("Iter: {}, D: {:.4}, G:{:.4}".format(it, d_loss[0], g_loss))

            it += 1

    # print("Final images")
    save_images(gen_imgs[:16], it)


def main():
    """Train pipeline"""
    from tensorflow.python.client import device_lib
    print("List local devices")
    print(device_lib.list_local_devices())

    print("\nGet available GPUs")
    print(K.tensorflow_backend._get_available_gpus())

    os.mkdir("/artefact/plots/")

    (x_train, y_train), (_, _) = load_mnist_data(dim=3)
    
    optimizer = Adam(0.0002, 0.5)
    
    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    n_disc_trainable = len(discriminator.trainable_weights)
    
    # Build the generator
    generator = build_generator(NOISE_DIM)
    n_gen_trainable = len(generator.trainable_weights)
    
    # The generator takes noise as input and generates imgs
    gen_inputs = Input(shape=(NOISE_DIM, ))
    img = generator(gen_inputs)
    
    # For the combined model we will only train the generator
    discriminator.trainable = False
    
    # The discriminator takes generated images as input and determines validity
    valid = discriminator(img)
    
    # The combined model (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(gen_inputs, valid)
    combined.compile(loss="binary_crossentropy", optimizer=optimizer)
    assert len(discriminator._collected_trainable_weights) == n_disc_trainable
    assert len(combined._collected_trainable_weights) == n_gen_trainable

    print("Training")
    start = time.time()
    train(generator, discriminator, combined, x_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS, noise_dim=NOISE_DIM)
    print("\tTime taken = {:.2f} mins".format((time.time() - start) / 60))

    print("Save Keras model")
    generator.save("/artefact/generator.h5")
    discriminator.save("/artefact/discriminator.h5")


if __name__ == "__main__":
    main()
