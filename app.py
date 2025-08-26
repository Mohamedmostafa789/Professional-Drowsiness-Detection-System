import streamlit as st
import cv2
import numpy as np
import time
from scipy.spatial.distance import euclidean
import mediapipe as mp


# --- Configuration ---
st.title("Professional Drowsiness Detector")
st.markdown("This app uses your webcam to detect drowsiness. A **visual alarm** will be displayed if your eyes are closed for too long.")

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15

# --- State Variables for Streamlit Session ---
if 'frames_closed' not in st.session_state:
    st.session_state.frames_closed = 0
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

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

    # Flip the frame horizontally for a "mirror" effect and convert to RGB
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # MediaPipe eye landmark indices for EAR calculation
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 374, 380]

            # Get the coordinates of the eye landmarks
            left_eye_points = np.array([(face_landmarks.landmark[i].x * img.shape[1],
                                         face_landmarks.landmark[i].y * img.shape[0])
                                         for i in left_eye_indices])
            right_eye_points = np.array([(face_landmarks.landmark[i].x * img.shape[1],
                                          face_landmarks.landmark[i].y * img.shape[0])
                                          for i in right_eye_indices])

            # Calculate the EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw contours around the eyes
            left_eye_hull = cv2.convexHull(left_eye_points.astype(np.int32))
            right_eye_hull = cv2.convexHull(right_eye_points.astype(np.int32))
            cv2.drawContours(img, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [right_eye_hull], -1, (0, 255, 0), 1)

            # Drowsiness detection logic
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
    bytes_data = picture.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    processed_img = process_image(img.copy())
    st.image(processed_img, channels="BGR", caption="Processed Image")
