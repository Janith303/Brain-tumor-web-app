import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enables communication between frontend and backend

# Define class labels matching your training configuration
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Path to your trained DenseNet121 model
MODEL_PATH = os.path.join('models', 'densenet121_best.keras')

print("Loading DenseNet121 model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server. Please check your models folder.'}), 500

    # Accept either 'image' or 'file' form keys from frontend
    file = request.files.get('image') or request.files.get('file')

    if not file or file.filename == '':
        return jsonify({'error': 'No valid image file provided in request'}), 400

    try:
        # Read file stream directly in memory
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))

        # Convert PIL image to array and format for DenseNet input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        # Apply DenseNet121 specific preprocessing
        processed_image = preprocess_input(img_array)

        # Perform inference
        predictions = model.predict(processed_image)
        print(f"DEBUG - Model output shape: {predictions.shape}")

        # Safety Check: Verify output shape matches 4-class classification (1, 4)
        if predictions.ndim != 2 or predictions.shape[1] != len(CLASS_NAMES):
            error_msg = (
                f"Model returned shape {predictions.shape}, but expected (1, 4). "
                f"The file in 'models/' is an intermediate layer or autoencoder "
                f"rather than the final classification model."
            )
            print(f"Prediction Error: {error_msg}")
            return jsonify({'error': error_msg}), 500

        # Extract predicted class index and confidence score
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return jsonify({
            'success': True,
            'class_name': predicted_class_name,
            'class': predicted_class_name,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)