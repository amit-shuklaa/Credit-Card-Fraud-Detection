from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load Model & Scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:  # If JSON request
            data = request.get_json()
            features = np.array(data["features"]).reshape(1, -1)
        else:  # If Form request
            features = [float(request.form[f"feature{i}"]) for i in range(1, 31)]
            features = np.array(features).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        result = "Fraudulent" if prediction == 1 else "Legitimate"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
