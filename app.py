import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "rain_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, prob=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        temp = float(request.form["Temperature"])
        hum = float(request.form["Humidity"])
        pres = float(request.form["Pressure"])
        wind = float(request.form["WindSpeed"])

        X = np.array([[temp, hum, pres, wind]])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])  # probability of Rain (class 1)

        result = "Rain Expected 🌧️" if pred == 1 else "No Rain Expected ☀️"
        return render_template("index.html", result=result, prob=prob, error=None)

    except Exception as e:
        return render_template("index.html", result=None, prob=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)