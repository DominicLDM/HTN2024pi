from threading import Thread
from flask import Flask, Response, send_from_directory, render_template
import cv2
import os
import requests
import io
import time
from ultralytics import YOLO

model = YOLO("./best.pt")

app = Flask(__name__)

# Initialize camera and face detection model (Haar Cascade)
cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Function to generate frames
def generate_frames():
    clip_number = 0
    start_time = time.time()
    out = cv2.VideoWriter(f'./outputs/output_{clip_number}.mp4', fourcc, 20.0, (frame_width, frame_height))
    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) == ord('q'):
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        # faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        results = model(frame)
        result = results[0]

        boxes = result.boxes.xyxy
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


        out.write(frame)
        
        if time.time() - start_time >= 10:
            out.release()
            thread = Thread(target=process_video, args={clip_number})
            thread.start()
              
            clip_number += 1
            start_time = time.time()
            out = cv2.VideoWriter(f'./outputs/output_{clip_number}.mp4', fourcc, 20.0, (frame_width, frame_height))

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame to create a stream
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    out.release()
    cap.release()
    cv2.destroyAllWindows

def process_video(clip_number):
    transcribed_text = transcribe(f'./outputs/output_{clip_number}.mp4')
    print(f'Transcription for clip {clip_number}: {transcribed_text}')


def transcribe(file_path):
    url = "https://symphoniclabs--symphonet-vsr-modal-htn-model-upload-static-htn.modal.run"

    with open(file_path, 'rb') as video_file:
        video = io.BytesIO(video_file.read())

    response = requests.post(url, files={'video': (file_path, video, 'video/mp4')})
    
    return response.text

# Route to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Serve static files (like HTML, CSS, JS)
@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(directory='static', path=path)

if __name__ == "__main__":


    app.run(host="0.0.0.0", port=8000)