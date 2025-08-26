import streamlit as st
import cv2
import numpy as np
import time
from scipy.spatial.distance import euclidean
# We've removed pygame, so we no longer need these imports.
# import pygame
# import os
import requests
from twilio.rest import Client
import mediapipe as mp

# --- IMPORTANT SETUP & TROUBLESHOOTING ---
# To fix the 'libGL.so.1' error on cloud services like Streamlit Cloud,
# you must create a file named `packages.txt` in the same directory
# as this script and add the following two lines to it:
# libgl1
# libglib2.0-0

# You also need a `requirements.txt` and `.streamlit/secrets.toml` file.
# The `requirements.txt` file should contain:
# streamlit
# opencv-python
# numpy
# scipy
# twilio
# mediapipe

# --- Configuration ---
st.title("Professional Drowsiness Detector")
st.markdown("This app uses your webcam to detect drowsiness. A **visual alarm** will be displayed and an SMS will be sent if your eyes are closed for too long.")

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15

# --- Twilio SMS Configuration (using Streamlit Secrets) ---
try:
    account_sid = st.secrets["twilio"]["account_sid"]
    auth_token = st.secrets["twilio"]["auth_token"]
    twilio_client = Client(account_sid, auth_token)
    
    # Replace these with your actual phone numbers
    TWILIO_PHONE_NUMBER = "+15017122661"  # Your Twilio phone number
    MY_PHONE_NUMBER = "+15558675310"      # The number to send the SMS to

    # Cooldown period for SMS alerts (in seconds)
    SMS_COOLDOWN = 300 # 5 minutes

except KeyError:
    st.error("Twilio API credentials not found in secrets.toml.")
    st.info("Please create a `.streamlit/secrets.toml` file with `account_sid` and `auth_token` under a `[twilio]` header.")
    twilio_client = None

# --- State Variables for Streamlit Session ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frames_closed' not in st.session_state:
    st.session_state.frames_closed = 0
# The alarm is now just a state, not a sound.
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False
if 'last_sms_sent' not in st.session_state:
    st.session_state.last_sms_sent = 0

# --- Visual Alarm and SMS Playback Functions ---
def activate_visual_alarm():
    """Activates the visual alarm."""
    st.session_state.alarm_active = True

def deactivate_visual_alarm():
    """Deactivates the visual alarm."""
    st.session_state.alarm_active = False

def send_sms_alert(message_body):
    """Sends an SMS alert using the Twilio client."""
    if twilio_client and (time.time() - st.session_state.last_sms_sent > SMS_COOLDOWN):
        try:
            message = twilio_client.messages.create(
                to=MY_PHONE_NUMBER,
                from_=TWILIO_PHONE_NUMBER,
                body=message_body
            )
            st.session_state.last_sms_sent = time.time()
            st.success(f"SMS alert sent! Message SID: {message.sid}")
        except Exception as e:
            st.warning(f"Failed to send SMS alert: {e}")

# --- Core Drowsiness Detection Function ---
def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a given eye."""
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame, face_mesh):
    # Flip the frame horizontally for a "mirror" effect
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # MediaPipe eye landmark indices for EAR calculation
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 374, 380]

            left_eye_points = np.array([(face_landmarks.landmark[i].x * frame.shape[1],
                                         face_landmarks.landmark[i].y * frame.shape[0])
                                         for i in left_eye_indices])
            right_eye_points = np.array([(face_landmarks.landmark[i].x * frame.shape[1],
                                          face_landmarks.landmark[i].y * frame.shape[0])
                                          for i in right_eye_indices])

            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw contours around the eyes
            left_eye_hull = cv2.convexHull(left_eye_points.astype(np.int32))
            right_eye_hull = cv2.convexHull(right_eye_points.astype(np.int32))
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            if avg_ear < EAR_THRESHOLD:
                st.session_state.frames_closed += 1
                if st.session_state.frames_closed >= CONSECUTIVE_FRAMES:
                    activate_visual_alarm()
                    cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    send_sms_alert("Drowsiness detected! Time to take a break.")
            else:
                st.session_state.frames_closed = 0
                deactivate_visual_alarm()
    else:
        st.session_state.frames_closed = 0
        deactivate_visual_alarm()
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# --- Main Streamlit App Logic ---
video_placeholder = st.empty()
status_message = st.empty()
visual_alarm_placeholder = st.empty()

if st.button('Start Detection'):
    st.session_state.is_running = True
    st.session_state.frames_closed = 0
    st.session_state.alarm_active = False
    
elif st.button('Stop Detection'):
    st.session_state.is_running = False
    deactivate_visual_alarm()

if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam. Please check permissions.")
        st.session_state.is_running = False
    else:
        status_message.info("Detection running. Looking for a face...")
        
        # Initialize MediaPipe FaceMesh
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    status_message.warning("Failed to get frame from webcam.")
                    break
                
                processed_frame = process_frame(frame.copy(), face_mesh)
                video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                
                if st.session_state.alarm_active:
                    # Display a large, red warning message
                    visual_alarm_placeholder.error("!!! DROWSINESS DETECTED !!!")
                else:
                    # Clear the message if the alarm is not active
                    visual_alarm_placeholder.empty()

                status_message.success(f"Eyes Open. Closed frames: {st.session_state.frames_closed}/{CONSECUTIVE_FRAMES}")


            cap.release()
            st.session_state.is_running = False
            deactivate_visual_alarm()
            status_message.info("Detection stopped.")

