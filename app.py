# Streamlit app for prediction
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from keras.models import load_model
model=load_model('Braintumor.h5')
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
def predict_tumor(image):
    # Resize the image
    image = cv2.resize(image, (150, 150))
    img_array = np.array(image)
    img_array = img_array.reshape(1, 150, 150, 3)

    # Predict the tumor type
    prediction = model.predict(img_array)
    indices = prediction.argmax()
    return labels[indices]

def main():
    st.title('Brain Tumor Detection')

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        tumor_type = predict_tumor(image)
        st.success(f'Tumor Type: {tumor_type}')

if __name__ == '__main__':
    main()
