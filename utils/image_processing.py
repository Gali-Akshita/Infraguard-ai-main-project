import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image):
    """
    Preprocess uploaded images using OpenCV:
    - Resize to 224x224
    - Normalize pixel values to 0-1
    """
    # Convert PIL Image to OpenCV format (numpy array)
    img_cv = np.array(image)
    
    # Handle RGBA images
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
        
    # Open CV uses BGR by default, but typically models are trained on RGB.
    # We will assume RGB model input, so no BGR conversion unless model strictly needs it.
    
    # Resize to 224x224
    img_resized = cv2.resize(img_cv, (224, 224))
    
    # Normalize pixel values to 0-1
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Expand dimensions to match model input shape (1, 224, 224, 3)
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    return img_expanded

def get_canny_edges(image: Image.Image):
    """
    Generate an edge detection visualization using OpenCV (Canny).
    """
    # Convert to RGB then to Grayscale
    img_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    return edges
