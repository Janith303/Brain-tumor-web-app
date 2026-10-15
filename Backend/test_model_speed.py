import time
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/densenet121_best.keras"

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

start = time.perf_counter()
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loading time:", round(time.perf_counter() - start, 2), "seconds")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# Test with one 224 x 224 RGB image
x = np.zeros((1, 224, 224, 3), dtype=np.float32)

# Warm-up
start = time.perf_counter()
_ = model(x, training=False).numpy()
print("Warm-up inference:", round(time.perf_counter() - start, 2), "seconds")

# Repeat the test three times
for i in range(3):
    start = time.perf_counter()
    _ = model(x, training=False).numpy()
    print(f"Run {i + 1}:", round(time.perf_counter() - start, 2), "seconds")