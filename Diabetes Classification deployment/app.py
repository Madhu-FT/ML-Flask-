import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    if prediction[0] == 1:
        result_text = "The person has diabetes."
    else:
        result_text = "The person does not have diabetes."

    return render_template("index.html", prediction_text=result_text)

if __name__ == "__main__":
    flask_app.run(debug=True)
