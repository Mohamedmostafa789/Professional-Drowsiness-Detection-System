<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Drowsiness Detector</title>
    <!-- Load Tailwind CSS for beautiful styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Load React and ReactDOM from CDNs -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <!-- Load Babel to transpile JSX in the browser -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <!-- Load face-api.js library from a CDN -->
    <script src="https://unpkg.com/face-api.js@0.22.2/dist/face-api.js"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #1a202c; /* Dark background from Tailwind */
        }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen flex items-center justify-center">

    <div id="root" class="w-full"></div>

    <script type="text/babel">
        // Get references to React's useState, useEffect, and useRef hooks
        const { useState, useEffect, useRef } = React;

        // The main component for our application
        const App = () => {
          // State variables to manage the application's UI and logic
          const [isModelsLoaded, setIsModelsLoaded] = useState(false);
          const [isDetectionRunning, setIsDetectionRunning] = useState(false);
          const [drowsinessMessage, setDrowsinessMessage] = useState('Detection is not running.');
          const [alarmActive, setAlarmActive] = useState(false);

          // References to HTML elements that we will interact with
          const videoRef = useRef(null);
          const canvasRef = useRef(null);

          // Configuration for the Eye Aspect Ratio (EAR)
          const EAR_THRESHOLD = 0.2;
          const EAR_CONSECUTIVE_FRAMES = 10; // Frames with eyes closed before alarm
          const [framesWithEyesClosed, setFramesWithEyesClosed] = useState(0);

          // A reference to the alarm sound's AudioContext
          const audioContextRef = useRef(null);
          const oscillatorRef = useRef(null);

          // Helper function to calculate Euclidean distance
          const euclideanDistance = (p1, p2) => {
            return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
          };

          // Function to calculate the Eye Aspect Ratio (EAR)
          const getEAR = (eye) => {
            const A = euclideanDistance(eye[1], eye[5]);
            const B = euclideanDistance(eye[2], eye[4]);
            const C = euclideanDistance(eye[0], eye[3]);
            const ear = (A + B) / (2.0 * C);
            return ear;
          };

          // Function to play a simple alarm sound
          const playAlarm = () => {
            if (audioContextRef.current === null) {
              audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (oscillatorRef.current === null) {
              const oscillator = audioContextRef.current.createOscillator();
              oscillator.type = 'sine';
              oscillator.frequency.value = 500;
              oscillator.connect(audioContextRef.current.destination);
              oscillator.start();
              oscillatorRef.current = oscillator;
            }
            setAlarmActive(true);
          };

          // Function to stop the alarm sound
          const stopAlarm = () => {
            if (oscillatorRef.current !== null) {
              oscillatorRef.current.stop();
              oscillatorRef.current.disconnect();
              oscillatorRef.current = null;
            }
            setAlarmActive(false);
          };

          // This useEffect hook runs once when the component mounts to load the face-api.js models
          useEffect(() => {
            const loadModels = async () => {
              try {
                setDrowsinessMessage('Loading models...');
                // Load the necessary pre-trained models.
                // NOTE: We are now using the more accurate SSD Mobilenet V1 model.
                await Promise.all([
                  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
                  faceapi.nets.faceLandmark68Net.loadFromUri('/models')
                ]);
                setIsModelsLoaded(true);
                setDrowsinessMessage('Models loaded. Click "Start Detection" to begin.');
              } catch (error) {
                console.error('Failed to load face-api models:', error);
                setDrowsinessMessage('Error: Failed to load models. Make sure the "models" folder is in the same directory as this file.');
              }
            };

            loadModels();
          }, []);

          // This useEffect handles the core detection loop
          useEffect(() => {
            let interval;
            const startDetection = async () => {
              if (!isModelsLoaded || !isDetectionRunning || !videoRef.current) {
                return;
              }

              if (videoRef.current.readyState < 4) {
                return;
              }

              const displaySize = { width: videoRef.current.width, height: videoRef.current.height };
              faceapi.matchDimensions(canvasRef.current, displaySize);

              interval = setInterval(async () => {
                // Use the new, more accurate SSD Mobilenet V1 face detector
                const detection = await faceapi.detectSingleFace(
                  videoRef.current,
                  new faceapi.SsdMobilenetv1Options()
                ).withFaceLandmarks();

                const ctx = canvasRef.current.getContext('2d');
                ctx.clearRect(0, 0, displaySize.width, displaySize.height);

                if (detection) {
                  const resizedDetections = faceapi.resizeResults(detection, displaySize);
                  faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
                  faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);

                  const landmarks = resizedDetections.landmarks;
                  const leftEye = landmarks.getLeftEye();
                  const rightEye = landmarks.getRightEye();

                  const leftEAR = getEAR(leftEye);
                  const rightEAR = getEAR(rightEye);
                  const avgEAR = (leftEAR + rightEAR) / 2.0;

                  if (avgEAR < EAR_THRESHOLD) {
                    setFramesWithEyesClosed(prev => prev + 1);
                    setDrowsinessMessage('Eyes are closed...');
                    if (framesWithEyesClosed >= EAR_CONSECUTIVE_FRAMES && !alarmActive) {
                      setDrowsinessMessage('Drowsiness detected! Wake up!');
                      playAlarm();
                    }
                  } else {
                    setFramesWithEyesClosed(0);
                    setDrowsinessMessage('Eyes are open. All clear.');
                    if (alarmActive) {
                      stopAlarm();
                    }
                  }
                } else {
                  setDrowsinessMessage('No face detected.');
                  setFramesWithEyesClosed(0);
                  if (alarmActive) {
                    stopAlarm();
                  }
                }
              }, 100);
            };

            startDetection();

            return () => clearInterval(interval);
          }, [isModelsLoaded, isDetectionRunning, framesWithEyesClosed, alarmActive]);

          // Function to start the webcam and detection
          const handleStartDetection = async () => {
            if (isDetectionRunning) {
              stopDetection();
              return;
            }
            
            if (!isModelsLoaded) {
              setDrowsinessMessage('Models are still loading. Please wait.');
              return;
            }

            try {
              const stream = await navigator.mediaDevices.getUserMedia({ video: true });
              if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                setDrowsinessMessage('Detection started. Looking for a face...');
                setIsDetectionRunning(true);
              }
            } catch (err) {
              console.error('Error accessing webcam:', err);
              setDrowsinessMessage('Error: Could not access webcam. Please check permissions.');
            }
          };

          // Function to stop the webcam and detection
          const stopDetection = () => {
            const stream = videoRef.current.srcObject;
            if (stream) {
              const tracks = stream.getTracks();
              tracks.forEach(track => track.stop());
            }
            if (alarmActive) {
              stopAlarm();
            }
            setIsDetectionRunning(false);
            setDrowsinessMessage('Detection stopped.');
          };

          return (
            <div className="flex flex-col items-center justify-center p-4 bg-gray-900 text-white min-h-screen font-inter">
              <div className="bg-gray-800 p-8 rounded-xl shadow-2xl w-full max-w-2xl mb-8">
                <h1 className="text-3xl font-bold mb-4 text-center text-indigo-400">
                  Professional Drowsiness Detector
                </h1>
                <p className="text-gray-400 text-center mb-6">
                  This app demonstrates real-time eye-closed detection using your webcam.
                  It sounds an alarm when drowsiness is detected.
                </p>
                <div className="relative w-full aspect-[4/3] rounded-lg overflow-hidden border-2 border-gray-700">
                  <video
                    ref={videoRef}
                    className="absolute top-0 left-0 w-full h-full object-cover"
                    width="640"
                    height="480"
                    muted
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full z-10"
                  />
                </div>
                <div className="mt-6 flex flex-col items-center">
                  <button
                    onClick={handleStartDetection}
                    disabled={!isModelsLoaded}
                    className={`
                      px-8 py-3 text-lg font-semibold rounded-full shadow-lg transition-all duration-300
                      ${isDetectionRunning
                        ? 'bg-red-600 hover:bg-red-700 text-white'
                        : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                      }
                      ${!isModelsLoaded ? 'opacity-50 cursor-not-allowed' : ''}
                    `}
                  >
                    {isDetectionRunning ? 'Stop Detection' : 'Start Detection'}
                  </button>
                  <div className="mt-4 text-lg font-medium text-center">
                    <p className={`${alarmActive ? 'text-red-500 font-bold' : 'text-gray-300'}`}>
                      {drowsinessMessage}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        };

        // Render the App component into the root div
        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>
