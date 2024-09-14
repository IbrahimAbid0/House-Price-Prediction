"""
This module contains unit tests for the Flask application.
It tests the functionality of the homepage, the prediction API,
and the model prediction.
"""

import unittest
from Feature-Frontend.app import app, model, preprocess_input


class FlaskTestCase(unittest.TestCase):
    """Test case for the Flask application."""

    def setUp(self):
        """Set up the Flask test client."""
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

    def test_homepage(self):
        """Test if the homepage is running correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        """Test the prediction API."""
        # Provide sample input data
        input_data = {
            'MSSubClass': 60,
            'MSZoning': 'RL',
            'LotArea': 8450,
            'LotConfig': 'Inside',
            'BldgType': '1Fam',
            'OverallCond': 5,
            'YearBuilt': 2003,
            'YearRemodAdd': 2003,
            'Exterior1st': 'VinylSd',
            'BsmtFinSF2': 0,
            'TotalBsmtSF': 856,
        }

        # Send POST request to the Flask app
        response = self.client.post('/', data=input_data)

        # Check if response is successful
        self.assertEqual(response.status_code, 200)
        # Check if the prediction output exists
        self.assertIn(b'Predicted Sale Price', response.data)

    def test_model_prediction(self):
        """Unit test for model prediction."""
        input_data = {
            'MSSubClass': 60,
            'MSZoning': 'RL',
            'LotArea': 8450,
            'LotConfig': 'Inside',
            'BldgType': '1Fam',
            'OverallCond': 5,
            'YearBuilt': 2003,
            'YearRemodAdd': 2003,
            'Exterior1st': 'VinylSd',
            'BsmtFinSF2': 0,
            'TotalBsmtSF': 856,
        }
        preprocessed_data = preprocess_input(input_data)
        prediction = model.predict(preprocessed_data)[0]

        # Check if the prediction is a float and within a reasonable range
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)  # Ensure the price is greater than 0


if __name__ == '__main__':
    unittest.main()
