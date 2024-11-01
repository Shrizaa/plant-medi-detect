import tensorflow as tf
import numpy as np
from PIL import Image

def clean_image(image):
    # Ensure the image is in RGB format and resize
    image = image.convert("RGB")  # Convert to RGB to remove any alpha channel
    image = image.resize((512, 512), Image.LANCZOS)  # Resize to the expected input shape
    return np.array(image)

def get_prediction(model, image):
    # Preprocess the image for prediction
    image = clean_image(image)  # Call clean_image to process the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale pixel values to [0, 1]
    
    predictions = model.predict(image)  # Make predictions
    predictions_arr = predictions[0]  # Get the first prediction
    return predictions, predictions_arr

def make_results(predictions, predictions_arr):
    # List of diseases corresponding to model output
    diseases = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']  
    max_index = np.argmax(predictions_arr)  # Get index of the highest prediction
    name = diseases[max_index]  # Get disease name
    status = "healthy" if max_index == 0 else "not healthy"  # Determine health status
    result = {
        "name": name,
        "status": status,
        "confidence": f"{int(predictions[0][max_index].round(2) * 100)}%"  # Format confidence
    }
    return result
