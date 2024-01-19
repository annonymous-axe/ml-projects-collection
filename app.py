import tensorflow as tf
import keras
import streamlit as st
from PIL import Image
import numpy as np

# Prediting function
def predict_img(img):

    new_model = tf.keras.models.load_model('CarsModelPrediction_Model_ResNet50.h5')

    img = np.array(Image.open(img))
    img = tf.reshape(img, [-1, img.shape[0], img.shape[1], img.shape[2]])
    y_proba = new_model.predict(img)  

    class_ = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']  

    car_name = class_[np.argmax(y_proba)]

    accuracy = (np.max(y_proba)*100).round(2)


    return car_name, accuracy


st.header("Car Company Classfication!")

st.write("This application clssified the given image into one of categroy from, ")
st.markdown(""" - Creta""")
st.markdown(""" - Audi""")
st.markdown(""" - Swift""")
st.markdown(""" - Rolls Royce""")
st.markdown(""" - Scorpio""")
st.markdown(""" - Tata Safari""")
st.markdown(""" - Innova""")

st.markdown("### Below upload the car image and press the 'Get Label' to check the car class.")
image = st.file_uploader(label="Upload car image", type=['png', 'jpg'])


if st.button("Get label"):
    if image is not None:
        st.image(image)

        with st.spinner("Predicting..."):
            name, score = predict_img(image)

        st.title(name)

        st.write(f'The model is {score} % sure!')
    else:
        st.warning("Please upload image first!")