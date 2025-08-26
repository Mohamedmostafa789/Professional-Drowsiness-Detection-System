# Professional Drowsiness Detection System

This project is a real-time drowsiness detection application that uses a person's webcam to monitor their eyes. If the user's eyes are closed for a specified number of consecutive frames, the system triggers an audible alarm to alert them.

## Features

- **Real-time Detection:** Uses a webcam to continuously monitor a user's face.
- **Eye Aspect Ratio (EAR):** Measures the opening and closing of the eyes to detect drowsiness.
- **Alarm System:** Plays an audible alarm (`mixkit-facility-alarm-sound-999.wav`) when drowsiness is detected.
- **Streamlit Interface:** Provides a simple and clean user interface with start/stop buttons.
- **Robustness:** Built with `dlib` for accurate facial landmark detection and `pygame` for reliable sound playback.

## Prerequisites

Before running the application, you need to download two external files and place them in the same directory as the script.

1.  **Dlib Facial Landmark Model:**
    - File Name: `shape_predictor_68_face_landmarks.dat`
    - Download Link: [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    - You must unzip this file after downloading.

2.  **Alarm Sound File:**
    - File Name: `mixkit-facility-alarm-sound-999.wav`
    - You should already have this file from our previous steps.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    ```
3.  **Activate the Virtual Environment:**
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
4.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Make sure you have downloaded the two prerequisite files and placed them in the project folder.
2.  From your terminal, with the virtual environment activated, run the Streamlit app:
    ```bash
    streamlit run drowsiness_detector.py
    ```
3.  Your default web browser will open a new tab with the application running.

## Contributing

Feel free to open issues or submit pull requests to improve this project.

## License

This project is licensed under the MIT License.
