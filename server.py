from flask import Flask, request, jsonify
from model.cnn_model import CnnModel  # Adjust the import based on your project structure
import numpy as np
import sys
from io import StringIO

app = Flask(__name__)
model_instance = CnnModel()

@app.route('/predict', methods=['POST', 'GET', 'OPTIONS'])
async def predict():
    try:
        # Get input data from the request
        input_data = request.json.get('input_data')

        # Load the model if not already loaded
        if not model_instance.model:
            await model_instance.load_model()

        # Redirect stdout to capture print statements
        original_stdout = sys.stdout
        sys.stdout = StringIO()

        # Perform prediction
        probability_vector = model_instance.predict(input_data)

        # Get the captured print statements
        prediction_stdout = sys.stdout.getvalue()

        print(prediction_stdout)
        # Restore the original stdout
        sys.stdout = original_stdout

        # Get the predicted label (index with maximum probability)
        predicted_label = np.argmax(probability_vector)
        predicted_label = f"{predicted_label}"

        # Respond with predictions
        return jsonify({
            'prediction_stdout': prediction_stdout,
            'probability_vector': probability_vector,
            'prediction': predicted_label})

    except Exception as error:
        print('Error:', str(error))
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(port=3000)
