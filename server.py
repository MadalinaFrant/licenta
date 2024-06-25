from flask import Flask, request, jsonify
from flask_cors import CORS

import joblib

from utils.utils import preprocess_text
from utils.utils import label_to_type


app = Flask(__name__)
CORS(app)

tfidf = joblib.load('machine_learning/models/TFIDF.pkl')
model = joblib.load('machine_learning/models/PAC.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not preprocess_text(text):
        return jsonify({'prediction': 'unknown'})

    tfidf_text = tfidf.transform([preprocess_text(text)])
    pred = model.predict(tfidf_text)
    prediction = pred[0]

    return jsonify({'prediction': label_to_type[prediction]})


if __name__ == '__main__':
    app.run(port=5000)

