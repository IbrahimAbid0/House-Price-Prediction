import unittest
import json
from app import app, model, preprocess_input  # Ensure you import your app and model

class FlaskTestCase(unittest.TestCase):
    # Test if Flask app is running correctly
    def test_homepage(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)

    # Test API for price prediction
    def test_prediction(self):
        tester = app.test_client(self)
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
        response = tester.post('/', data=input_data)

        # Check if response is successful
        self.assertEqual(response.status_code, 200)
        
        # Check if the prediction output exists
        # self.assertIn(b'price', response.data)
        self.assertIn(b'Predicted Sale Price', response.data)


    # Unit test for the model
    def test_model_prediction(self):
        # Sample preprocessed data (you'll have to adjust based on your model requirements)
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
