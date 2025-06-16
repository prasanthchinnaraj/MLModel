from flask import Flask, request, render_template
import joblib
import numpy as np

# Load trained model
model = joblib.load('linear_house_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    prediction = model.predict(np.array([[sqft]]))
    return render_template('index.html', prediction_text=f'Predicted House Price: ₹{prediction[0]:.2f} Lakhs')

if __name__ == '__main__':
    app.run(debug=True)
    