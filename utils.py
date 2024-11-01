import tensorflow as tf
import numpy as np
from PIL import Image

def clean_image(image):
    # Resize and preprocess the image as needed
    image = image.resize((512, 512), Image.LANCZOS)
    return np.array(image)

def get_prediction(model, image):
    # Preprocess the image for prediction
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale pixel values
    predictions = model.predict(image)
    predictions_arr = predictions[0]
    return predictions, predictions_arr
def make_results(predictions, predictions_arr):
    # Update this list with the correct disease names
    diseases = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']  
    max_index = np.argmax(predictions_arr)
    name = diseases[max_index]
    status = "healthy" if max_index == 0 else "not healthy"
    result = {
        "name": name,
        "status": status,
        "confidence": f"{int(predictions[0][max_index].round(2) * 100)}%"
    }
    return result
