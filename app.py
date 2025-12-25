from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("scaler.joblib")

# Feature columns (same order as training)
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

@app.route("/")
def home():
    return "Diabetes Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Convert JSON input to DataFrame
    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    else:
        input_df = pd.DataFrame(data)

    # Ensure correct column order
    input_df = input_df[FEATURE_COLUMNS]

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Make prediction
    predictions = model.predict(input_scaled)

    return jsonify({
        "predictions": predictions.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
