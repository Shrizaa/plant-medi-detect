import tensorflow as tf
import numpy as np
from PIL import Image

def clean_image(image):
    """
    Preprocess the input image for prediction by resizing it to the expected size
    and converting it to RGB format.
    
    Args:
        image (PIL.Image): The input image in PIL format.
    
    Returns:
        np.ndarray: The processed image as a NumPy array with shape (512, 512, 3).
    """
    image = image.convert("RGB")  # Convert to RGB to remove any alpha channel
    image = image.resize((512, 512), Image.LANCZOS)  # Resize to the expected input shape
    image = np.array(image)  # Convert image to numpy array
    
    # Ensure correct shape for the model
    if image.shape[-1] == 4:  # If there are 4 channels (RGBA)
        image = image[..., :3]  # Take only the first three channels (RGB)
    
    return image

def get_prediction(model, image):
    """
    Get the prediction for the input image from the trained model.
    
    Args:
        model (tf.keras.Model): The trained TensorFlow model.
        image (PIL.Image): The input image in PIL format.
    
    Returns:
        tuple: Predictions and predictions array.
    """
    image = clean_image(image)  # Call clean_image to process the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale pixel values to [0, 1]
    
    predictions = model.predict(image)  # Make predictions
    predictions_arr = predictions[0]  # Get the first prediction
    return predictions, predictions_arr

def make_results(predictions, predictions_arr):
    """
    Format the results from the predictions into a readable format.
    
    Args:
        predictions (np.ndarray): The raw predictions from the model.
        predictions_arr (np.ndarray): The array of prediction probabilities.
    
    Returns:
        dict: A dictionary containing the predicted disease name, health status, and confidence.
    """
    diseases = ['Healthy', 'Multiple Diseases', 'Rust', 'Scab']  # Update this list with the correct disease names
    max_index = np.argmax(predictions_arr)  # Get index of the highest prediction
    name = diseases[max_index]  # Get disease name
    status = "healthy" if max_index == 0 else "not healthy"  # Determine health status
    result = {
        "name": name,
        "status": status,
        "confidence": f"{int(predictions[0][max_index].round(2) * 100)}%"  # Format confidence
    }
    return result

# Example usage (make sure to load your model first)
if __name__ == "__main__":
    # Load your trained model here
    model = tf.keras.models.load_model('path_to_your_model.h5')  # Replace with your model path
    
    # Load an image for prediction
    image_path = 'path_to_your_image.jpg'  # Replace with your image path
    image = Image.open(image_path)  # Load the image using PIL
    
    # Get predictions
    predictions, predictions_arr = get_prediction(model, image)
    
    # Format results
    result = make_results(predictions, predictions_arr)
    
    # Output results
    print(result)
