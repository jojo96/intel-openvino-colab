import streamlit as st
import cv2
from PIL import Image, ImageDraw, ImageFont

st.header("Intel Openvino Demo")
st.write("Power of inference with Intel software. Select a model on the left and click See Demo.")
 
Options = [" ","vehicle-detection demo1","vehicle-detection demo2","face-landmark-detection demo1","face-landmark-detection demo2"]
choose = st.sidebar.selectbox("Pick a model:", Options)

if st.sidebar.button('See Demo'):

    if choose == "vehicle-detection demo1":
        #Options2 = [" ","image1","image2"]
        #choose2 = st.selectbox("Select image:", Options2)
        st.write("This model presents a vehicle attributes classification algorithm for a traffic analysis scenario.")
        image = Image.open('results/vehicle-attributes-recognition-barrier-0039/car.jpg')
        image2 = Image.open('results/vehicle-attributes-recognition-barrier-0039/car1.png')
        st.image(image, caption='Original', use_column_width=True)
        st.image(image2, caption='Inference', use_column_width=True)  

    if choose == "vehicle-detection demo2":
        #Options2 = [" ","image1","image2"]
        #choose2 = st.selectbox("Select image:", Options2)
        st.write("This model presents a vehicle attributes classification algorithm for a traffic analysis scenario.")
        image = Image.open('results/vehicle-attributes-recognition-barrier-0039/truck.jpg')
        image2 = Image.open('results/vehicle-attributes-recognition-barrier-0039/truck1.jpg')
        st.image(image, caption='Original', use_column_width=True)
        st.image(image2, caption='Inference', use_column_width=True) 

    if choose == "face-landmark-detection demo1":
        #Options2 = [" ","image1","image2"]
        #choose2 = st.selectbox("Select image:", Options2)
        st.write("The model predicts five facial landmarks: two eyes, nose, and two lip corners.")
        image = Image.open('results/landmarks-regression-retail-0009/angry.png')
        image2 = Image.open('results/landmarks-regression-retail-0009/angry1.png')
        st.image(image, caption='Original')
        st.image(image2, caption='Inference')   
 
    if choose == "face-landmark-detection demo2":
        #Options2 = [" ","image1","image2"]
        #choose2 = st.selectbox("Select image:", Options2)
        st.write("The model predicts five facial landmarks: two eyes, nose, and two lip corners.")
        image = Image.open('results/landmarks-regression-retail-0009/smile.png')
        image2 = Image.open('results/landmarks-regression-retail-0009/smile1.png')
        st.image(image, caption='Original')
        st.image(image2, caption='Inference')    
