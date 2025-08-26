import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.spatial.distance import euclidean

# --- IMPORTANT SETUP ---
# You need to have an alarm sound file: 'mixkit-facility-alarm-sound-999.wav'
# in the same directory as this script.
#
# Your `requirements.txt` file should contain the following:
# streamlit
# opencv-python
# mediapipe
# scipy
# numpy

# --- Configuration ---
st.title("Real-Time Drowsiness Detector")
st.markdown("This app uses your live webcam feed to detect drowsiness. An alarm will sound if your eyes are closed for too long.")

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15
ALARM_SOUND_FILE = "mixkit-facility-alarm-sound-999.wav"

# --- MediaPipe Setup ---
# Initialize MediaPipe's face mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get the indexes for the left and right eye landmarks from the face mesh model
# These are fixed indexes based on the MediaPipe model documentation
RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# --- State Variables for Streamlit Session ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frames_closed' not in st.session_state:
    st.session_state.frames_closed = 0
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# --- Core Drowsiness Detection Function ---
def eye_aspect_ratio(eye_landmarks):
    """Calculates the Eye Aspect Ratio (EAR) for a given eye."""
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame):
    """Processes a single frame for drowsiness detection."""
    # To improve performance, optionally mark the frame as not writeable to pass by reference.
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Re-enable writeable flag
    frame.flags.writeable = True
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for both eyes
            left_eye = np.array([(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in [33, 163, 144, 145, 153, 154]])
            right_eye = np.array([(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in [362, 382, 381, 380, 374, 373]])
            
            # Draw eye contours (optional, for visualization)
            cv2.drawContours(frame_bgr, [np.int32(left_eye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_bgr, [np.int32(right_eye)], -1, (0, 255, 0), 1)

            # Calculate EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Check for drowsiness
            if avg_ear < EAR_THRESHOLD:
                st.session_state.frames_closed += 1
                if st.session_state.frames_closed >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame_bgr, "!!! DROWSINESS DETECTED !!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    st.session_state.alarm_active = True
            else:
                st.session_state.frames_closed = 0
                st.session_state.alarm_active = False
    else:
        st.session_state.frames_closed = 0
        st.session_state.alarm_active = False
        cv2.putText(frame_bgr, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame_bgr

# --- Main Streamlit App Logic ---
video_placeholder = st.empty()
status_message = st.empty()
# A placeholder for the audio player
audio_placeholder = st.empty()

col1, col2 = st.columns(2)
with col1:
    if st.button('Start Detection'):
        st.session_state.is_running = True
        st.session_state.frames_closed = 0
        st.session_state.alarm_active = False
        st.rerun()

with col2:
    if st.button('Stop Detection'):
        st.session_state.is_running = False
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.rerun()

if st.session_state.is_running:
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)
    
    if not st.session_state.cap.isOpened():
        st.error("Error: Could not access webcam. Please check permissions.")
        st.session_state.is_running = False
    else:
        status_message.info("Detection running. Looking for a face...")
        
        ret, frame = st.session_state.cap.read()
        if not ret:
            status_message.warning("Failed to get frame from webcam.")
            st.session_state.cap.release()
            st.session_state.cap = None
        else:
            frame = cv2.flip(frame, 1)
            processed_frame = process_frame(frame.copy())
            
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            if st.session_state.alarm_active:
                status_message.error("!!! DROWSINESS DETECTED! Wake up! !!!")
                # Use st.audio to play the sound directly in the browser
                if os.path.exists(ALARM_SOUND_FILE):
                    audio_placeholder.audio(ALARM_SOUND_FILE, format="audio/wav", autoplay=True, loop=True, key='alarm_audio')
            else:
                status_message.success(f"Eyes Open. Closed frames: {st.session_state.frames_closed}/{CONSECUTIVE_FRAMES}")
                # Clear the audio placeholder when the alarm is not active
                audio_placeholder.empty()

        st.rerun()

else:
    if 'cap' in st.session_state and st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
    status_message.info("Detection stopped.")
