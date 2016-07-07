import numpy as np
import unittest
import project.modelling.train_model as modelling
from keras.datasets import mnist
from mockredis import MockRedis

class TrainModelTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1337)
        self.data = mnist.load_data()
        self.redis = MockRedis()

    def test_transform_data(self):
        """Test that the shape are correct after transformation."""
        X_train, X_test, y_train, y_test = modelling.transform_data(self.data, 10)
        self.assertEqual(X_train.shape[1], 784)
        self.assertEqual(X_test.shape[1], 784)
        self.assertEqual(y_train.shape[1], 10)
        self.assertEqual(y_test.shape[1], 10)

    def test_evaluate_model_accuracy(self):
        """Test if the accuracy is correct."""
        X_train, X_test, y_train, y_test = modelling.transform_data(self.data, 10)
        results, model = modelling.evaluate_model(X_train, X_test, y_train,
                                                  y_test, 128, 1)
        self.assertAlmostEqual(results[1], 0.96970000000000001)

    def test_save_model_network(self):
        """Test if the architecture is saved."""
        X_train, X_test, y_train, y_test = modelling.transform_data(self.data, 10)
        results, model = modelling.evaluate_model(X_train, X_test, y_train,
                                                  y_test, 128, 1)
        modelling.save_model(model, self.redis)
        self.assertEqual(self.redis.keys()[1].decode("UTF-8"), "model")

    def test_save_model_weights(self):
        """Test if the weights are stored correctly."""
        X_train, X_test, y_train, y_test = modelling.transform_data(self.data, 10)
        results, model = modelling.evaluate_model(X_train, X_test, y_train,
                                                  y_test, 128, 1)
        modelling.save_model(model, self.redis)
        self.assertEqual(self.redis.keys()[0].decode("UTF-8"), "weights")
