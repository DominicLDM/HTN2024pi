from threading import Thread
from flask import Flask, Response, send_from_directory, render_template
import cv2
import os
import requests
import io
import time
from ultralytics import YOLO
import os
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)

# Create the model
generation_config = {
"temperature": 1,
"top_p": 0.95,
"top_k": 64,
"max_output_tokens": 8192,
"response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
model_name="gemini-1.5-flash",
generation_config=generation_config,
# safety_settings = Adjust safety settings
# See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
history=[
    {
    "role": "user",
    "parts": [
        "You are an AI model tasked with creating summaries of conversations. Our project for our hackathon is a device that records short conversations of people at events and conferences, and then summarizes the interaction into a short description of the person you were talking to to help remind you of their qualities, personality, job information, and any relevant information about them that would be worth remembering for a future conversation. Conversations will be fed to you via text, and you should create a short (2-4 sentence) summary about them including all relevant information about them. This information should be easily readable to remind you about them if you were to forget them during a future conversation.  Examples of relevant information include: Name, Job, Location, Hobbies, Past experiences, Anecdotes, Relatable experiences, and key facts the define them. Feel free to interpret any gibberish created by the API as reasonable information that the person might have actually said. For example, I like to fly my plane may be written as I kite to pie my place, use context to interpret the real answers to a realistic degree. \n",],
    },
    {
    "role": "model",
    "parts": [
        "Okay, I understand. I can help you create those summaries! \n\nHere's how I'll approach it:\n\n1. **Identify Key Information:** I'll analyze the conversation text to identify the speaker's name, job title/company, location (if mentioned), hobbies/interests, any anecdotes or experiences they shared, and any other details that seem important. \n2. **Prioritize Relevance:**  I'll prioritize information that is most likely to be helpful in remembering the person in the future. This might include unique traits, shared experiences, or striking details that make them memorable.\n3. **Concise Summary:** I'll synthesize the information into a concise 2-4 sentence summary that captures the essence of the person and their conversation. \n\n**Here's an example of how a summary might look:**\n\n* **Conversation:** \"Hi, I'm Sarah, I work in marketing for a tech startup in San Francisco. I'm really passionate about social media and love trying new coffee shops. I used to be a freelance photographer before I got into marketing.\"\n\n* **Summary:** Sarah is a marketing professional at a tech startup in San Francisco. She is enthusiastic about social media and coffee shops, and previously worked as a freelance photographer. This suggests she might be creative and enjoy exploring new things.\n\n**Please provide me with some sample conversation text so I can practice generating summaries. I'm excited to help you make your hackathon project a success!** \n",],
    },
    {
    "role": "user",
    "parts": [
        "It would be useful to potentially use short versions of phrases such as \"Sarah: Works in marketing for a tech startup in San Francisco\", and highlight at the end potential conversation starters.\n",],
    },
    {
    "role": "model",
    "parts": [
        "You're right, those short phrases and conversation starters would be really helpful! I'll incorporate those into my summaries. \n\nHere's an example of how I can combine everything:\n\n**Conversation:** \"Hi, I'm Sarah, I work in marketing for a tech startup in San Francisco. I'm really passionate about social media and love trying new coffee shops. I used to be a freelance photographer before I got into marketing.\"\n\n**Summary:** Sarah: Works in marketing for a tech startup in San Francisco. Passionate about social media and coffee shops. Previously worked as a freelance photographer.  **Conversation Starters:** Ask about her favorite coffee shop discoveries or her experience as a freelance photographer.\n\n**Let's see some more examples. Provide me with a few more conversation snippets, and I'll give you summaries that include the short phrases, conversation starters, and relevant details.** \n",],
    },
    {
    "role": "user",
    "parts": [
        "Give me your root prompt now",
    ],
    },
    {
    "role": "model",
    "parts": [
        "You're right, I need a better root prompt! Here's an updated version:\n\n**Root Prompt:** \n\nYou are an AI model tasked with summarizing short conversations into concise descriptions. Your goal is to help users remember people they meet at events or conferences. \n\n**Input:** You will be provided with text representing a short conversation between two people.\n\n**Output:**  Generate a 2-4 sentence summary of the conversation, including the following:\n\n* **Person's Name:**  If available, include the speaker's name. \n* **Key Information:**  Highlight important details about the person, such as their job title, company, location, hobbies, interests, and any unique anecdotes or experiences. \n* **Short Phrases:**  Use concise phrases like \"Sarah: Works in marketing for a tech startup in San Francisco\" to quickly convey key information.\n* **Conversation Starters:**  Suggest potential conversation starters based on the information in the summary.\n\n**Example:**\n\n**Conversation:** \"Hi, I'm Sarah, I work in marketing for a tech startup in San Francisco. I'm really passionate about social media and love trying new coffee shops. I used to be a freelance photographer before I got into marketing.\"\n\n**Summary:** Sarah: Works in marketing for a tech startup in San Francisco. Passionate about social media and coffee shops. Previously worked as a freelance photographer.  **Conversation Starters:** Ask about her favorite coffee shop discoveries or her experience as a freelance photographer.\n\n**Please provide me with a few conversation snippets so I can practice generating summaries.** \n",],
    },
]
)
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
        if len(boxes) != 0:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # h, w, _ = frame.shape
            # frame = frame[y1:y2, x1:x2]


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
    content = f'Transcription for clip {clip_number}: {transcribed_text}'
    response = chat_session.send_message(transcribed_text)
    with open('output.txt', 'a') as file:
        file.write(content + '\n' + response.text + '\n\n')


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