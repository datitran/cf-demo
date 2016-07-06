import os
import unittest
import project.prediction.prediction_app as prediction_app
from project.prediction.prediction_app import app
from unittest.mock import patch
from keras.models import model_from_json
from keras.optimizers import RMSprop

def mock_get_model(redis=None):
    model = model_from_json(open(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "data/mnist_mlp.json")).read())
    model.load_weights(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "data/mnist_mlp_weights.h5"))
    model.compile(loss="categorical_crossentropy",
              optimizer=RMSprop())
    return model

@patch("project.prediction.prediction_app.get_model", mock_get_model)
class PredictionAppTest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.files = {"files": open(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "data/four_test.png"), "rb")}
        self.image = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "data/four_test.png")

    def test_main_page(self):
        """Test that the status code 200 is returned for get."""
        results = self.app.get("/")
        self.assertEqual(results.status_code, 200)

    def test_convert_image(self):
        """Test if the image was converted correctly."""
        converted_image = prediction_app.convert_image(self.image)
        self.assertAlmostEqual(converted_image.sum(), 46.129411764)

    def test_prediction_status(self):
        """Test that the status code 200 is returned for post."""
        results = self.app.post("/prediction", data=self.files)
        self.assertEqual(results.status_code, 200)

    def test_prediction_results(self):
        """Test that the right prediction is returned."""
        results = self.app.post("/prediction", data=self.files)
        self.assertEqual(results.get_data(as_text=True), "4")
