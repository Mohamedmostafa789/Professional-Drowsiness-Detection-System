import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# --- IMPORTANT: SETUP INSTRUCTIONS ---
# 1. You must have a 'requirements.txt' file with the following lines:
#    streamlit
#    opencv-python
#    numpy
#    scipy
#    mediapipe
#    streamlit-webrtc
#    av
#
# 2. You must have a 'packages.txt' file with the following lines to install
#    the necessary system dependencies on the Streamlit Cloud server:
#    libgl1
#    libglib2.0-0
#    
# This version avoids the single-frame limitation of st.camera_input() by using a
# real-time video streamer. It also removes the audio alarm to fix PyAudio installation issues.

# --- Configuration ---
st.title("Real-Time Drowsiness Detector")
st.markdown("This app uses your live webcam feed to detect drowsiness. A **visual alarm** will be displayed when your eyes are closed for too long.")

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15

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

class VideoProcessor(VideoTransformerBase):
    """Processes video frames in real-time to detect drowsiness."""
    def __init__(self):
        # Initialize state variables
        self.frames_closed = 0
        self.alarm_active = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Receives a video frame, processes it, and returns the modified frame.
        This method is called for every frame in the video stream.
        """
        img = frame.to_ndarray(format="bgr24")

        # Flip the frame for a mirror effect and convert to RGB
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
                    self.frames_closed += 1
                    if self.frames_closed >= CONSECUTIVE_FRAMES:
                        self.alarm_active = True
                else:
                    self.frames_closed = 0
                    self.alarm_active = False
        else:
            self.frames_closed = 0
            self.alarm_active = False
            cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display status text based on alarm state
        if self.alarm_active:
            cv2.putText(img, "!!! DROWSINESS DETECTED !!!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Eyes Open. All Clear.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main Streamlit App Logic ---
webrtc_ctx = webrtc_streamer(
    key="drowsiness-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

if webrtc_ctx.state.playing:
    st.write("Detection is running...")
