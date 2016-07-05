from project.prediction.prediction_app import app
import unittest
import os
from PIL import Image, ImageOps
import numpy as np

class PredictionAppTest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def tearDown(self):
        pass

    def test_main_page(self):
        results = self.app.get("/")
        self.assertEqual(results.status_code, 200)

    def test_prediction(self):
        results = self.app.get("/prediction")
        self.assertEqual(results.status_code, 200)
