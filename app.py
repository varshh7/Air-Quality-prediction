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
    try:
        # Extract all 13 input features from the form
        data = [float(request.form[f'feature{i+1}']) for i in range(13)]
        
        # Make the prediction
        prediction = model.predict([data])
        
        # Determine the prediction result (you can change this to your specific interpretation of the model's output)
        result = 'Above Average' if prediction[0] else 'Below Average'
        
        # Return the result along with the form, using render_template
        return render_template('index.html', prediction=result)
    
    except Exception as e:
        # In case of error, render the form with an error message
        return render_template('index.html', prediction=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
