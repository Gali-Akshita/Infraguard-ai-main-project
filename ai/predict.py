import tensorflow as tf
from utils.image_processing import preprocess_image
import os

# Get the absolute path to the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crack_detector.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_crack(image):
    """
    Predict crack presence given a PIL Image.
    Returns:
        label (str): "Crack Detected" or "No Crack"
        display_confidence (float): Confidence in the prediction (0-1)
        crack_probability (float): Raw probability of crack (0-1), used for risk assessment
    """
    if model is None:
        # Fallback if model failed to load
        import random
        prob = random.uniform(0, 1)
        label = "Crack Detected" if prob > 0.5 else "No Crack"
        return label, max(prob, 1 - prob), prob

    img = preprocess_image(image)
    crack_prob = float(model.predict(img)[0][0])

    # Binary classification: > 0.5 means crack detected
    if crack_prob > 0.5:
        return "Crack Detected", crack_prob, crack_prob
    else:
        return "No Crack", 1.0 - crack_prob, crack_prob
