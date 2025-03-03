import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# App title
st.title("Face Detection App")

# Instructions
st.markdown("""
### How to Use:
1. Enable the camera.
2. Adjust the detection parameters (`scaleFactor` and `minNeighbors`).
3. Choose the color of the rectangle to highlight detected faces.
4. Click the 'Detect Faces' button.
5. Download the processed image if desired.
""")

# Start camera
camera = st.camera_input("Take a picture")

if camera is not None:
    image = Image.open(camera)
    img_array = np.array(image)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Parameter selection
    scaleFactor = st.slider("Scale Factor", 1.1, 2.0, 1.1, 0.1)
    minNeighbors = st.slider("Min Neighbors", 1, 10, 5)
    color = st.color_picker("Pick a rectangle color", "#FF0000")
    color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))  # Convert hex to BGR
    
    if st.button("Detect Faces"):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 2)
        
        st.image(img_array, caption="Detected Faces", use_column_width=True)
        
        # Save the processed image
        save_path = "detected_faces.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        with open(save_path, "rb") as file:
            st.download_button("Download Image", file, file_name="detected_faces.jpg", mime="image/jpeg")
