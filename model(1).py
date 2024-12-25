# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:53:59 2024
@author: admin
"""

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load('LogisticRegression.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Données envoyées sous forme JSON
    features = data['features']  # Exemple : {"features": [1.0, 2.0, 3.0]}
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
