import tensorflow
import cv2
import streamlit as st
import time
import numpy as np

st.title("     --------Sign detection--------")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
model=tensorflow.keras.models.load_model(r"C:\Users\nithya\Downloads\DeepVisionModel.h5")

font = cv2.FONT_HERSHEY_SIMPLEX     
org = (0, 30)       
fontScale = 0.5
color = (255, 0, 0)     
thickness = 2

while run:
    _, frame1 = camera.read()
    image_resize = cv2.resize(frame1,(256,256))
    frame1 = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
    time.sleep(0.05)
    _, frame2 = camera.read()
    image_resize = cv2.resize(frame2,(256,256))
    frame2 = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
    time.sleep(0.05)
    image_1_b_w = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY )
    image_2_b_w = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY )
    absdiff = cv2.absdiff(image_1_b_w,image_2_b_w)
    
    nabsdiff=np.dstack([absdiff,absdiff,absdiff])

    FRAME_WINDOW.image(nabsdiff)
    abd=np.expand_dims(nabsdiff,axis=0)
    op=model.predict(abd)
    if op==1:
        absdiff = cv2.putText(nabsdiff, 'Signed', org, font,fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW.image(absdiff)

    else:
        absdiff = cv2.putText(nabsdiff, 'UnSigned', org, font,fontScale, color, thickness, cv2.LINE_AA)
        FRAME_WINDOW.image(absdiff)
else:
    st.write('Stopped')
