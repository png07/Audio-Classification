import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Paths to model, label encoder, and image folder
MODEL_PATH = r'C:\Users\pratik\Documents\Projects\audio_classification\savedmodels\final_model_l90.keras'
LABELENCODER_PATH = r'C:\Users\pratik\Documents\Projects\audio_classification\savedmodels\label_encoder_l90.pkl'
IMAGE_FOLDER = r'C:\Users\pratik\Documents\Projects\audio_classification\images'

# Load model and label encoder
model = load_model(MODEL_PATH)
with open(LABELENCODER_PATH, 'rb') as file:
    labelencoder = pickle.load(file)

def predict_audio_label(audio_data, sample_rate):
    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features, axis=1)

    # Reshape for Conv1D model
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, 40, 1)  # Adding the channel dimension

    # Predict using the trained Conv1D model
    y_pred = model.predict(mfccs_scaled_features)

    # Get the predicted class
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Decode the predicted class label
    prediction_class = labelencoder.inverse_transform(y_pred_classes)

    return prediction_class[0]

# Streamlit app
st.title("Audio Classification")

# File uploader for .wav and .mp3 formats
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3","mpeg","ogg"])

if uploaded_file is not None:
    
    # Load the audio file into librosa from the uploaded buffer
    audio_data, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast') 

    # Predict the label
    predicted_label = predict_audio_label(audio_data, sample_rate)

    # Display the predicted label
    st.markdown(f"<h3>Predicted Label: {predicted_label}</h3>", unsafe_allow_html=True)

    st.audio(uploaded_file, format='audio/wav') 
    
    # Display the corresponding image
    image_path = os.path.join(IMAGE_FOLDER, f"{predicted_label}.jpg")
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, width=400)
    else:
        st.write("No image found for the predicted label.")
