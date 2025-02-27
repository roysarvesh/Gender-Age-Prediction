import cv2
import numpy as np
import time
import threading
import argparse

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, faceBoxes

def predict_age_gender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    gender = genderList[genderNet.forward()[0].argmax()]
    
    ageNet.setInput(blob)
    age = ageList[ageNet.forward()[0].argmax()]
    
    return gender, age

def process_frame(frame):
    frame, faceBoxes = detect_face(frame)
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                     max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]
        gender, age = predict_age_gender(face)
        cv2.putText(frame, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def real_time_detection():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        start_time = time.time()
        frame = process_frame(frame)
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Age & Gender Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
