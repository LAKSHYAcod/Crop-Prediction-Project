import numpy as np
from flask import Flask, request, render_template
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("crop.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print(request.form)
        input_features = [
            float(request.form["nitrogen"]),
            float(request.form["Phosphorus"]),
            float(request.form["potassium"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["pH"]),
            float(request.form["rainfall"])
        ]
        scaled_input = scaler.transform([input_features])

        prediction_encoded = model.predict(scaled_input)
        predicted_label = encoder.inverse_transform(prediction_encoded)[0]

        return render_template("crop.html", Prediction_block=predicted_label)

    except Exception as e:
        return render_template("crop.html", Prediction_block=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)