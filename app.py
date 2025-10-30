from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('cancer_detection_model.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from HTML form
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(final_input)
        result = "Cancer (Positive)" if prediction[0][0] > 0.5 else "No Cancer (Negative)"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)