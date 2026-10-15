import os
import io
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Available CPUs:", tf.config.list_physical_devices('CPU'))

app = Flask(__name__)
CORS(app)  # Enables communication between frontend and backend

# Class labels: must match the order used in training
# (check with train_gen.class_indices or train_ds.class_names)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

IMG_SIZE = (224, 224)

# Change this to the exact filename of the model you renamed
MODEL_PATH = os.path.join('models', 'densenet121_best.keras')

# ---------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------
print("Loading DenseNet121 model...")
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Input shape:", model.input_shape, "| Output shape:", model.output_shape)

    # Fail early if this is the wrong file
    assert model.output_shape[-1] == len(CLASS_NAMES), (
        f"Wrong model file: output shape is {model.output_shape}, "
        f"expected (None, {len(CLASS_NAMES)})"
    )

    # Warm-up: the first call is slow because TensorFlow builds its graph.
    # Doing it here means the first real request is already fast.
    _ = model(np.zeros((1, *IMG_SIZE, 3), dtype=np.float32), training=False)
    print("Model loaded and warmed up successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok' if model is not None else 'model_not_loaded'})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server. Please check your models folder.'}), 500

    # Accept either 'image' or 'file' form keys from frontend
    file = request.files.get('image') or request.files.get('file')

    if not file or file.filename == '':
        return jsonify({'error': 'No valid image file provided in request'}), 400

    try:
        t0 = time.time()

        # Read the upload in memory and prepare it
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize(IMG_SIZE)

        img_array = np.asarray(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # DenseNet121 preprocessing.
        # Remove this line if your model already has a Rescaling/preprocessing
        # layer inside it, or if you trained with plain /255.0.
        processed_image = img_array

        t1 = time.time()

        # Direct call is faster than model.predict() for a single image
        predictions = model(processed_image, training=False).numpy()

        t2 = time.time()
        print(f"Preprocess: {t1 - t0:.3f}s | Inference: {t2 - t1:.3f}s")

        # Safety check: output must be (1, 4)
        if predictions.ndim != 2 or predictions.shape[1] != len(CLASS_NAMES):
            error_msg = (
                f"Model returned shape {predictions.shape}, but expected (1, {len(CLASS_NAMES)}). "
                f"The loaded file is not the final classification model."
            )
            print(f"Prediction Error: {error_msg}")
            return jsonify({'error': error_msg}), 500

        probs = predictions[0]
        predicted_class_index = int(np.argmax(probs))
        confidence = float(probs[predicted_class_index])
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        return jsonify({
            'success': True,
            'class_name': predicted_class_name,
            'class': predicted_class_name,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                name: round(float(p) * 100, 2) for name, p in zip(CLASS_NAMES, probs)
            }
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # debug=False avoids the reloader loading the model twice.
    # Set debug=True only while editing code.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)