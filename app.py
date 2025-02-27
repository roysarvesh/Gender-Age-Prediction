import cv2
import numpy as np
import streamlit as st
from PIL import Image

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def detect_face(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faceBoxes.append([x1, y1, x2, y2])
    return faceBoxes

def predict_age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    gender = genderList[genderNet.forward()[0].argmax()]
    
    ageNet.setInput(blob)
    age = ageList[ageNet.forward()[0].argmax()]
    
    return gender, age

def process_image(image):
    frame = np.array(image)
    faceBoxes = detect_face(frame)
    result_text = []
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                     max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]
        gender, age = predict_age_gender(face)
        result_text.append(f'{gender}, {age}')
    return result_text

st.title("Age & Gender Prediction")

option = st.radio("Choose Input Mode", ("Upload Image", "Real-time Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        results = process_image(image)
        for res in results:
            st.write(res)

elif option == "Real-time Webcam":
    st.write("Starting Webcam... Click 'Stop' to exit")
    cap = cv2.VideoCapture(0)
    stop = st.button("Stop")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break
        faceBoxes = detect_face(frame)
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                         max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]
            gender, age = predict_age_gender(face)
            cv2.rectangle(frame, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        st.image(frame, channels="BGR")
    
    cap.release()
    cv2.destroyAllWindows()
