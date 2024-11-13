from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract all 13 input features from the form
    try:
        data = [float(request.form[f'feature{i+1}']) for i in range(13)]
        prediction = model.predict([data])
        result = 'Above Average' if prediction[0] else 'Below Average'
        return jsonify({'Prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

