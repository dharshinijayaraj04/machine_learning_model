from flask import Flask, request, jsonify
import joblib
import pandas as pd
app = Flask(__name__)
# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

print("Libraries imported and model loaded successfully.")



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Convert the input data into a pandas DataFrame
    # X_train is available from the previous steps, we can use its columns for ordering
    # Ensure data is a list of dicts if it's a single dict to create a DataFrame
    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    else:
        input_df = pd.DataFrame(data)

    # Ensure the columns are in the same order as the training data
    # X_train.columns needs to be accessible here, so assuming X_train is globally available
    # However, in a real Flask app, you might need to save X_train columns or mock it.
    # Using the columns from the original X dataframe (which represents all features)
    input_df = input_df[X.columns]

    # Make prediction
    predictions = model.predict(input_df)

    # Convert predictions to a list of integers
    predictions_list = predictions.tolist()

    # Return the prediction result as a JSON response
    return jsonify({"predictions": predictions_list})

if __name__ == '__main__':
    app.run(debug=True)

print("Flask application run configuration added.")
