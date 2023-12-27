from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        values = [
    float(request.form['mean_radius']),
    float(request.form['mean_texture']),
    float(request.form['mean_perimeter']),
    float(request.form['mean_area']),
    float(request.form['mean_smoothness'])
]
        columns = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
        df = pd.DataFrame([values], columns=columns)
        df_scaled=StandardScaler().fit_transform(df)
        print(df_scaled)
        try:
            pred = model.predict(df_scaled)
            return render_template('result.html', prediction=pred[0])
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message)  
            return render_template('result.html', prediction=error_message)

if __name__ == '_main_':
    app.run(debug=True)