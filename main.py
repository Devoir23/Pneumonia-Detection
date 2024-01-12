# Import necessary libraries
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained VGG16 model
model = load_model("Pneumonia.h5")

# Define the class labels
classes = ['Normal','PNEUMONIA']

# Streamlit app
st.title("Pneumonia Detection")
st.header("Upload an Image")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # # Display the uploaded image
    # image = Image.open(uploaded_file).convert('RGB')
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    #
    # # Preprocess the image for model prediction
    # resized_img = cv2.resize(np.array(image), (224, 224))  # Resize image to the expected input size
    # i = np.array(resized_img) / 255.0
    # i = np.expand_dims(i, axis=0)
    # img_data = preprocess_input(i)
    #
    # # Make prediction using the loaded model
    # prediction = model.predict(img_data)
    # predicted_class_index = np.argmax(prediction, axis=1)
    #
    # if predicted_class_index[0] == 0:
    #     st.write("Person is Affected By PNEUMONIA")
    # else:
    #     st.write("Result is Normal")

    #new:
    # Read the image file with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Convert the image from BGR (OpenCV default) to RGB
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # Display the uploaded image (convert back to PIL format for Streamlit)
    display_image = Image.fromarray(opencv_image)
    st.image(display_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for model prediction
    resized_img = cv2.resize(opencv_image, (224, 224))  # Resize image to the expected input size
    i = resized_img / 255.0
    i = np.expand_dims(i, axis=0)
    # img_data = preprocess_input(i)
    # Make prediction using the loaded model
    prediction = model.predict(i)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]

    st.write(f"The predicted tumor type is: {predicted_class} \n\n accuracy: {prediction[0][predicted_class_index]:.2f}")
