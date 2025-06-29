import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
# from tensorflow.keras.models import model_from_json

# Load model once
model = load_model('model/vehicle_model2.h5')

# # Load architecture
# with open("model.json", "r") as json_file:
#     loaded_model_json = json_file.read()

# model = model_from_json(loaded_model_json)

# Load weights
# model.load_weights("model_weights.weights.h5")

vehicle_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory='Vehicle_image/Vehicles',
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)
class_names = vehicle_data.class_names
st.write(class_names)

# Streamlit UI
st.title("Vehicle Type Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])  

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")
    # try:
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption="Uploaded Image")
    # except Exception as e:
    #     st.error(f"Error opening image: {e}")

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    # if st.button("Predict"):
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = 100 * predictions[0][predicted_index]

    # st.write("Prediction probabilities:", predictions[0])
    st.subheader(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
