from flask import Flask, render_template, request
import joblib  # To load your trained model and encoder
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load your trained model and encoder (adjust paths accordingly)
model = joblib.load('linear_regression_model.pkl')  # Replace with the correct path
encoder = joblib.load('onehot_encoder.pkl')  # OneHotEncoder from your notebook

# Preprocessing function to transform input
def preprocess_input(data):
    """Preprocess the input data for prediction."""
    df = pd.DataFrame([data])

    # List of categorical columns used in your notebook
    categorical_cols = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']
    
    # Ensure categorical columns are correctly handled by the encoder
    df_encoded = pd.DataFrame(encoder.transform(df[categorical_cols]), index=df.index)
    df_encoded.columns = encoder.get_feature_names_out(categorical_cols)

    # Drop original categorical columns and concatenate the encoded ones
    df.drop(categorical_cols, axis=1, inplace=True)
    df_final = pd.concat([df, df_encoded], axis=1)
    
    return df_final

# Flask route to handle prediction
@app.route("/", methods=["GET", "POST"])
def index():
    """Handle the prediction request and render the result."""
    price = None
    if request.method == "POST":
        # Collect input data from form
        data = {
            'MSSubClass': int(request.form['MSSubClass']),
            'MSZoning': request.form['MSZoning'],
            'LotArea': int(request.form['LotArea']),
            'LotConfig': request.form['LotConfig'],
            'BldgType': request.form['BldgType'],
            'OverallCond': int(request.form['OverallCond']),
            'YearBuilt': int(request.form['YearBuilt']),
            'YearRemodAdd': int(request.form['YearRemodAdd']),
            'Exterior1st': request.form['Exterior1st'],
            'BsmtFinSF2': int(request.form['BsmtFinSF2']),
            'TotalBsmtSF': int(request.form['TotalBsmtSF']),
        }

        # Preprocess input using the updated function
        preprocessed_data = preprocess_input(data)

        # Make prediction using the loaded model
        prediction = model.predict(preprocessed_data)[0]

        # Format the prediction as price
        price = f"{prediction:,.2f}"

    # Render HTML page with prediction
    return render_template('index.html', price=price)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
