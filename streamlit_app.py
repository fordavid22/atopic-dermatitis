import base64, io
import json
import requests

import streamlit as st
from PIL import Image

URL = "http://127.0.0.1:5000/detect_skin_defect"

st.set_page_config(layout="wide", page_title="Atopic Eczema Classifier")

st.write("## Check if your skin defect is atopic eczema or not.")
st.write(
            """
            Upload or take picture of the affected area of your skin and see if the defect
            is an atopic eczema or not.
            """
        )

st.sidebar.write("## Upload or take picture from your camera :gear:")

if "uploaded_img_changed" not in st.session_state:
    st.session_state.uploaded_img_changed = 0

if "camera_img_changed" not in st.session_state:
    st.session_state.camera_img_changed = 0

def uploaded_on_change_callback():
    st.session_state.uploaded_img_changed = 1
    st.session_state.camera_img_changed = 0

def camera_on_change_callback():
    st.session_state.uploaded_img_changed = 0
    st.session_state.camera_img_changed = 1

def convert_image(img):
    buffer = io.BytesIO()
    img.save(buffer, format="png")
    byte_img = base64.b64encode(buffer.getvalue()).decode()


    return byte_img

def camera_img_callback(img_input): 
    # img_input = st.session_state.camera_img
    img = Image.open(img_input) 
    img_col.write("Camera Input Image :camera:")
    img_col.image(img)
    encoded_img = base64.b64encode(img_input.getvalue()).decode()
    classify_image(encoded_img)

def file_upload_callback(img_input): 
    # img_input = st.session_state.uploaded_img
    img = Image.open(img_input) 
    img_col.write("Uploaded Input Image :camera:")
    img_col.image(img)

    encoded_img = convert_image(img)
    classify_image(encoded_img)

def classify_image(encoded_img_str):
    body = {"image":encoded_img_str}
    response = requests.post(url=URL, json=body)
    if response.status_code == 200:
        result = response.json()
        st.info(
                    f"""
                        {result['predicted_classes'][0]}: {result['probabilities'][0] * 100:0.2f}%

                        {result['predicted_classes'][1]}: {result['probabilities'][1] * 100:0.2f}%
                    """
                )
    else:
        st.error("Could not perform classification.")

img_col = st.columns(1)[0]
uploaded_img = st.sidebar.file_uploader(
                label="Upload and image", type=["png", "jpg", "jpeg"],
                help="Ensure that the uploaded image is focused on the area of the affected skin.",
                on_change=uploaded_on_change_callback
            )
camera_img = st.sidebar.camera_input(
                label="Take picture of your skin",
                help="Ensure the center of the image is on the area of the affected skin.",
                on_change=camera_on_change_callback
            )

if st.session_state.uploaded_img_changed and uploaded_img:
    file_upload_callback(uploaded_img)
elif st.session_state.camera_img_changed and camera_img:
    camera_img_callback(camera_img)
