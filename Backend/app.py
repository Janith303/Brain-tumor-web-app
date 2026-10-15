import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image

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
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save temporary file
        temp_path = os.path.join('temp.jpg')
        file.save(temp_path)

        # Load and resize image to 224x224 (matching training specs)
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 224, 224, 3)

        # Apply DenseNet121 specific preprocessing (no double scaling)
        processed_image = preprocess_input(img_array)

        # Perform prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'class_name': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        if os.path.exists('temp.jpg'):
            os.remove('temp.jpg')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)