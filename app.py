import streamlit as st
import cv2
import dlib
from scipy.spatial.distance import euclidean
import numpy as np
import time

# --- IMPORTANT: SETUP INSTRUCTIONS ---
# 1. You must have a 'requirements.txt' file with the following lines:
#    streamlit
#    opencv-python
#    dlib
#    scipy
#    numpy
#
# 2. You must have a 'packages.txt' file with the following lines to install
#    the necessary system dependencies on the Streamlit Cloud server:
#    libgl1
#    libglib2.0-0
#
# 3. Download the facial landmark model from dlib:
#    - Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#    - Unzip it to get 'shape_predictor_68_face_landmarks.dat'
#    - Place this .dat file in the same directory as this Python script.

# --- Configuration ---
st.title("Professional Drowsiness Detector")
st.markdown("This app uses your webcam to detect drowsiness. A **visual alarm** will be displayed if your eyes are closed for too long.")

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15

# --- Dlib Setup ---
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    st.error(f"Error loading dlib model: {e}")
    st.info("Please make sure you have downloaded 'shape_predictor_68_face_landmarks.dat' and placed it in the same folder.")
    st.stop()

# Get the indexes for the left and right eye landmarks from the 68-point model
(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

# --- State Variables for Streamlit Session ---
if 'frames_closed' not in st.session_state:
    st.session_state.frames_closed = 0
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False

# --- Core Drowsiness Detection Functions ---
def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a given eye."""
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_image(img):
    """Processes a single image for drowsiness detection."""
    global frames_closed

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(lStart, lEnd)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(rStart, rEnd)])
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        cv2.drawContours(img, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if avg_ear < EAR_THRESHOLD:
            st.session_state.frames_closed += 1
            if st.session_state.frames_closed >= CONSECUTIVE_FRAMES:
                st.session_state.alarm_active = True
                cv2.putText(img, "!!! DROWSINESS DETECTED !!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            st.session_state.frames_closed = 0
            st.session_state.alarm_active = False
    else:
        st.session_state.frames_closed = 0
        st.session_state.alarm_active = False
        cv2.putText(img, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

# --- Main Streamlit App Logic ---
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    picture = st.camera_input("Take a picture to analyze...")
    
with col2:
    if st.session_state.alarm_active:
        st.error("!!! Drowsiness Detected! Wake up! !!!")
    else:
        st.success(f"Eyes Open. Closed frames: {st.session_state.frames_closed}/{CONSECUTIVE_FRAMES}")

# Process the image if one is provided
if picture:
    # Read the image from the camera input as a NumPy array
    bytes_data = picture.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Process the image and display the result
    processed_img = process_image(img.copy())
    st.image(processed_img, channels="BGR", caption="Processed Image")
