import os
import numpy as np
np.random.seed(1337)
import redis
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

r = redis.StrictRedis(host="redis.local.pcfdev.io", port=55098, password="c2aa5c33-9099-4d37-a27f-fbb8af882e2d")

NB_CLASSES = 10
BATCH_SIZE = 128
NB_EPOCH = 20

def transform_data(data, nb_classes):
    """Reshape/transform the MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = data
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, y_train, y_test

def evaluate_model(X_train, X_test, y_train, y_test, batch_size, nb_epoch):
    """Returns loss/accuracy and the model."""
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop(),
              metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, validation_data=(X_test, y_test))
    results = model.evaluate(X_test, y_test, verbose=0)
    return results, model

def save_model(model, redis):
    json_string = model.to_json()
    redis.set("model", json_string)
    model.save_weights("mnist_mlp.h5")
    with open("mnist_mlp.h5", "rb") as f:
        redis.set("weights", f.read())
    os.remove("mnist_mlp.h5")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = transform_data(mnist.load_data(),
            NB_CLASSES)

    results, model = evaluate_model(X_train, X_test, y_train, y_test,
                                    BATCH_SIZE, NB_EPOCH)

    save_model(model, r)
