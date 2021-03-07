from tensorflow.keras.utils import to_categorical


def load_mnist_dataset():
    from tensorflow.keras.datasets import mnist
    # load dataset
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # reshape dataset to have a single channel
    train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
    test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
    # one hot encode target values
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    return train_x, train_y, test_x, test_y
