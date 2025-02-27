Gender & Age Prediction

This project is a real-time gender and age prediction system using OpenCV and Streamlit. It allows face detection and age/gender classification on images and live webcam feed using deep learning models.
Features

Face Detection – Detects faces in images and live video streams.
Age & Gender Prediction – Uses deep learning models to predict age group and gender.
Real-time Processing – Supports real-time prediction via webcam.
User-Friendly Interface – Streamlit-based UI for image and webcam input.
Two Implementation Modes –

    app.py (Streamlit Web App) – Upload an image or use a webcam.
    detect.py (OpenCV Script) – OpenCV-based real-time processing.

Installation
1. Clone the Repository

git clone https://github.com/roysarvesh/Gender-And-Age-Prediction.git
cd Gender-And-Age-Prediction

2. Install Dependencies

pip install -r requirements.txt

Libraries Used (with Versions)
Library	Version
opencv-python	4.x.x
numpy	1.x.x
streamlit	1.x.x
Pillow	9.x.x

    Note: Versions may vary. Check your installed versions using pip list.

Usage
Run the Streamlit App

To launch the Streamlit-based UI for image and webcam-based prediction:

streamlit run app.py

Run the OpenCV-based Script

For real-time detection using OpenCV:

python detect.py

Required Model Files

Download the following pre-trained model files and place them in the project directory:

    opencv_face_detector.pbtxt
    opencv_face_detector_uint8.pb
    age_deploy.prototxt
    age_net.caffemodel
    gender_deploy.prototxt
    gender_net.caffemodel

Controls

    For app.py (Streamlit Web App) – Click the Stop button to exit real-time mode.
    For detect.py (OpenCV Script) – Press 'Q' to quit the webcam window.
![NoteGPT-Flowchart-1740660427966](https://github.com/user-attachments/assets/60cdce42-2228-49b6-8dc7-3b3b550f7c86)


Author

👤 roysarvesh
🔗 GitHub: github.com/roysarvesh
Contributions & Issues

Feel free to open an issue or contribute with a pull request!
