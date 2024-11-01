# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Load the Model and cache it
@st.cache_data
def load_model(path):
    # Define the Xception model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Define the DenseNet model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensemble the models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.load_weights(path)  # Load model weights
    return model

# Streamlit app layout
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Hide Streamlit menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Load the model
model = load_model('model.h5')

# Title and description
st.title('Plant Disease Detection')
st.write("Upload your plant's leaf image to get predictions on whether the plant is healthy or diseased.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# Process uploaded file
if uploaded_file is not None:
    # Display progress and text
    progress_text = st.text("Processing Image...")
    my_bar = st.progress(0)

    # Read the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    resized_image = image.resize((700, 400), Image.LANCZOS)  # Use LANCZOS for resizing
    st.image(resized_image, caption='Uploaded Image', use_column_width=True)
    my_bar.progress(30)

    # Clean the image
    try:
        cleaned_image = clean_image(image)
        my_bar.progress(60)

        # Make predictions
        predictions, predictions_arr = get_prediction(model, cleaned_image)
        my_bar.progress(90)

        # Interpret the results
        result = make_results(predictions, predictions_arr)

        # Show the results
        st.write(f"The plant is **{result['status']}** with a prediction of **{result['prediction']}**.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

    # Reset progress bar
    my_bar.empty()
    progress_text.empty()


