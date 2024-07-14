from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load(open('best_model.pkl', 'rb')) 
scaler = joblib.load(open('scaler.pkl', 'rb'))  

features = ['Quantity', 'UnitPrice', 'Sales', 'Sales_rolling_mean', 'Sales_pct_change', 'Sales_lag']

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST', 'GET'])
def predict():
        
        input_data = [
            float(request.form['Quantity']),
            float(request.form['UnitPrice']),
            float(request.form['Sales']),
            float(request.form['Sales_rolling_mean']),
            float(request.form['Sales_pct_change']),
            float(request.form['Sales_lag'])
        ]

        # Create a DataFrame from the input data
        df = pd.DataFrame([input_data], columns=features)
        
        # Scale the input data
        X_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        result = {'anomaly': bool(prediction)}
        
        return render_template('index.html', result=result)
    

if __name__ == '__main__':
    app.run(debug=True)
