# import streamlit as st
# import cv2
# import dlib
# from scipy.spatial.distance import euclidean
# import numpy as np
# import pygame  # Importing the new library
# from threading import Thread
# import os

# # --- IMPORTANT SETUP ---
# # You need to download two things for this code to work:
# # 1. The dlib model: 'shape_predictor_68_face_landmarks.dat'
# #    - Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# #    - Unzip it and place the .dat file in the same directory as this script.
# # 2. The audio file: 'mixkit-facility-alarm-sound-999.wav'
# #    - Make sure this file is in the same directory as this script.
# #
# # You also need to install the required Python libraries.
# # The 'playsound' library caused an error, so we'll use 'pygame' instead.
# # To install 'pygame', run the following command in your terminal:
# # pip install pygame streamlit opencv-python dlib numpy

# # --- Configuration ---
# st.title("Professional Drowsiness Detector")
# st.markdown("This app uses your webcam to detect drowsiness. An alarm will sound if your eyes are closed for too long.")

# # Parameters for drowsiness detection
# EAR_THRESHOLD = 0.25  # The threshold for the Eye Aspect Ratio (EAR)
# CONSECUTIVE_FRAMES = 15  # Number of consecutive frames eyes must be closed to trigger the alarm

# # The path to the alarm sound file. Make sure it's in the same folder as the script.
# ALARM_SOUND_FILE = "mixkit-facility-alarm-sound-999.wav"

# # --- Dlib Setup ---
# try:
#     # Initialize dlib's face detector and the facial landmark predictor
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# except RuntimeError as e:
#     st.error(f"Error loading dlib model: {e}")
#     st.info("Please make sure you have downloaded 'shape_predictor_68_face_landmarks.dat' and placed it in the same folder.")
#     st.stop()

# # Get the indexes for the left and right eye landmarks
# (lStart, lEnd) = (36, 42)
# (rStart, rEnd) = (42, 48)

# # --- State Variables for Streamlit Session ---
# if 'is_running' not in st.session_state:
#     st.session_state.is_running = False
# if 'frames_closed' not in st.session_state:
#     st.session_state.frames_closed = 0
# if 'alarm_active' not in st.session_state:
#     st.session_state.alarm_active = False

# # --- Pygame Sound Setup ---
# try:
#     pygame.mixer.init()
#     if os.path.exists(ALARM_SOUND_FILE):
#         alarm_sound = pygame.mixer.Sound(ALARM_SOUND_FILE)
#     else:
#         st.error(f"Alarm sound file '{ALARM_SOUND_FILE}' not found. Please place it in the same folder.")
#         st.stop()
# except pygame.error as e:
#     st.error(f"Error initializing pygame mixer: {e}")
#     st.stop()

# # --- Sound Playback Functions ---
# def play_alarm():
#     """Plays the alarm sound using pygame in a loop."""
#     if not st.session_state.alarm_active:
#         st.session_state.alarm_active = True
#         alarm_sound.play(-1)  # -1 makes the sound loop indefinitely

# def stop_alarm():
#     """Stops the alarm sound."""
#     if st.session_state.alarm_active:
#         st.session_state.alarm_active = False
#         pygame.mixer.stop()

# # --- Core Drowsiness Detection Function ---
# def eye_aspect_ratio(eye):
#     """Calculates the Eye Aspect Ratio (EAR) for a given eye."""
#     A = euclidean(eye[1], eye[5])
#     B = euclidean(eye[2], eye[4])
#     C = euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def process_frame(frame):
#     """Processes a single frame for drowsiness detection."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray, 0)
    
#     if len(faces) > 0:
#         face = faces[0]
#         landmarks = predictor(gray, face)
        
#         left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(lStart, lEnd)])
#         right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(rStart, rEnd)])
        
#         left_ear = eye_aspect_ratio(left_eye)
#         right_ear = eye_aspect_ratio(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0
        
#         cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

#         (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
#         if avg_ear < EAR_THRESHOLD:
#             st.session_state.frames_closed += 1
#             if st.session_state.frames_closed >= CONSECUTIVE_FRAMES:
#                 cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 play_alarm() # Call the new alarm function
#         else:
#             st.session_state.frames_closed = 0
#             stop_alarm()
#     else:
#         st.session_state.frames_closed = 0
#         stop_alarm()
#         cv2.putText(frame, "No face detected", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     return frame

# # --- Main Streamlit App Logic ---
# video_placeholder = st.empty()
# status_message = st.empty()

# # Create a start/stop button for the webcam
# if st.button('Start Detection'):
#     st.session_state.is_running = True
#     st.session_state.frames_closed = 0
#     st.session_state.alarm_active = False
    
# elif st.button('Stop Detection'):
#     st.session_state.is_running = False
#     stop_alarm()

# # Main loop for video processing
# if st.session_state.is_running:
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Could not access webcam. Please check permissions.")
#         st.session_state.is_running = False
#     else:
#         status_message.info("Detection running. Looking for a face...")
        
#         while st.session_state.is_running:
#             ret, frame = cap.read()
#             if not ret:
#                 status_message.warning("Failed to get frame from webcam.")
#                 break
            
#             frame = cv2.flip(frame, 1)
#             processed_frame = process_frame(frame.copy())
#             video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
#             if st.session_state.alarm_active:
#                 status_message.error("!!! Drowsiness Detected! Wake up! !!!")
#             else:
#                 status_message.success(f"Eyes Open. Closed frames: {st.session_state.frames_closed}/{CONSECUTIVE_FRAMES}")

#         cap.release()
#         st.session_state.is_running = False
#         status_message.info("Detection stopped.")
import streamlit as st
import cv2
import dlib
from scipy.spatial.distance import euclidean
import numpy as np
import pygame
import os
import requests # Added for downloading the model file
import bz2 # Added for decompressing the model file
import sys

# --- IMPORTANT SETUP ---
# This app now automatically downloads the required dlib model file.
# You still need to ensure your 'mixkit-facility-alarm-sound-999.wav' file is in the same directory.
#
# You also need to install the required Python libraries.
# To install them, run the following command in your terminal:
# pip install streamlit opencv-python dlib numpy pygame requests bz2-python

# --- Configuration ---
st.title("Professional Drowsiness Detector")
st.markdown("This app uses your webcam to detect drowsiness. An alarm will sound if your eyes are closed for too long.")

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25  # The threshold for the Eye Aspect Ratio (EAR)
CONSECUTIVE_FRAMES = 15  # Number of consecutive frames eyes must be closed to trigger the alarm

# The path to the alarm sound file. Make sure it's in the same folder as the script.
ALARM_SOUND_FILE = "mixkit-facility-alarm-sound-999.wav"

# --- Dlib Model Auto-Download and Decompression ---
# This function handles the downloading of the large dlib model file.
DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DLIB_MODEL_COMPRESSED_FILENAME = "shape_predictor_68_face_landmarks.dat.bz2"
DLIB_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"

def download_dlib_model(url, filename):
    """
    Downloads a file from a given URL and saves it with the specified filename.
    Includes a progress bar for a better user experience.
    """
    st.info(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        downloaded_size = 0
        progress_bar = st.progress(0)
        with open(filename, "wb") as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded_size += len(data)
                progress = min(downloaded_size / total_size, 1.0)
                progress_bar.progress(progress)
        st.success("Download complete!")

    except requests.exceptions.RequestException as e:
        st.error(f"Error: Failed to download the dlib model file. Please check your internet connection. Details: {e}")
        st.stop()

# Check if the uncompressed file already exists.
if not os.path.exists(DLIB_MODEL_FILENAME):
    st.warning(f"Model file not found: {DLIB_MODEL_FILENAME}")
    
    # Check if the compressed file exists
    if not os.path.exists(DLIB_MODEL_COMPRESSED_FILENAME):
        download_dlib_model(DLIB_MODEL_URL, DLIB_MODEL_COMPRESSED_FILENAME)

    # Decompress the file
    st.info(f"Decompressing {DLIB_MODEL_COMPRESSED_FILENAME}...")
    try:
        with open(DLIB_MODEL_COMPRESSED_FILENAME, 'rb') as source_file:
            compressed_data = source_file.read()
        decompressed_data = bz2.decompress(compressed_data)
        with open(DLIB_MODEL_FILENAME, 'wb') as dest_file:
            dest_file.write(decompressed_data)
        st.success("Decompression complete!")
        # Optional: remove the compressed file after decompression
        os.remove(DLIB_MODEL_COMPRESSED_FILENAME)
        st.info(f"Removed compressed file: {DLIB_MODEL_COMPRESSED_FILENAME}")

    except Exception as e:
        st.error(f"Error during decompression: {e}")
        st.stop()

else:
    st.info(f"Model file already exists: {DLIB_MODEL_FILENAME}")

# --- Dlib Setup ---
try:
    # Initialize dlib's face detector and the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_MODEL_FILENAME) # Now uses the variable for the file path
except RuntimeError as e:
    st.error(f"Error loading dlib model: {e}")
    st.info("There was a problem initializing the dlib predictor. Please ensure the model file is not corrupted.")
    st.stop()

# Get the indexes for the left and right eye landmarks
(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

# --- State Variables for Streamlit Session ---
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frames_closed' not in st.session_state:
    st.session_state.frames_closed = 0
if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False

# --- Pygame Sound Setup ---
try:
    pygame.mixer.init()
    if os.path.exists(ALARM_SOUND_FILE):
        alarm_sound = pygame.mixer.Sound(ALARM_SOUND_FILE)
    else:
        st.error(f"Alarm sound file '{ALARM_SOUND_FILE}' not found. Please place it in the same folder.")
        st.stop()
except pygame.error as e:
    st.error(f"Error initializing pygame mixer: {e}")
    st.stop()

# --- Sound Playback Functions ---
def play_alarm():
    """Plays the alarm sound using pygame in a loop."""
    if not st.session_state.alarm_active:
        st.session_state.alarm_active = True
        alarm_sound.play(-1)  # -1 makes the sound loop indefinitely

def stop_alarm():
    """Stops the alarm sound."""
    if st.session_state.alarm_active:
        st.session_state.alarm_active = False
        pygame.mixer.stop()

# --- Core Drowsiness Detection Function ---
def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a given eye."""
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame):
    """Processes a single frame for drowsiness detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(lStart, lEnd)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(rStart, rEnd)])
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_ear)
        avg_ear = (left_ear + right_ear) / 2.0
        
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if avg_ear < EAR_THRESHOLD:
            st.session_state.frames_closed += 1
            if st.session_state.frames_closed >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_alarm()
        else:
            st.session_state.frames_closed = 0
            stop_alarm()
    else:
        st.session_state.frames_closed = 0
        stop_alarm()
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# --- Main Streamlit App Logic ---
video_placeholder = st.empty()
status_message = st.empty()

# Create a start/stop button for the webcam
if st.button('Start Detection'):
    st.session_state.is_running = True
    st.session_state.frames_closed = 0
    st.session_state.alarm_active = False
    
elif st.button('Stop Detection'):
    st.session_state.is_running = False
    stop_alarm()

# Main loop for video processing
if st.session_state.is_running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam. Please check permissions.")
        st.session_state.is_running = False
    else:
        status_message.info("Detection running. Looking for a face...")
        
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                status_message.warning("Failed to get frame from webcam.")
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = process_frame(frame.copy())
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            if st.session_state.alarm_active:
                status_message.error("!!! Drowsiness Detected! Wake up! !!!")
            else:
                status_message.success(f"Eyes Open. Closed frames: {st.session_state.frames_closed}/{CONSECUTIVE_FRAMES}")

        cap.release()
        st.session_state.is_running = False
        status_message.info("Detection stopped.")

