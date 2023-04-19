import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from threading import Thread, Lock
import queue
import time

app = Flask(__name__)
socketio = SocketIO(app)

known_face_encodings = []
known_face_names = ["Ben Capper", "Bill Gates", "Jeff Bezos"]
image_filenames = ["images/ben.jpg", "images/bill.jpg", "images/jeff.jpg"]

for filename in image_filenames:
    img = face_recognition.load_image_file(filename)
    face_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(face_encoding)


frame_queue = queue.Queue()
processed_frames_queue = queue.Queue()
lock = Lock()

def read_frames(camera):
    desired_fps = 15  # Desired framerate
    interval = 1.0 / desired_fps  # Time between each frame

    while True:
        start_time = time.time()

        ret, frame = camera.read()
        if not ret:
            break

        frame_queue.put(frame)

        # Sleep for the remaining time in the interval
        time_to_next_frame = interval - (time.time() - start_time)
        if time_to_next_frame > 0:
            time.sleep(time_to_next_frame)


def process_frames():
    frame_count = 0
    frame_skip = 2  # Process one frame out of three
    box_offset = 40  # Adjust this value to change the box size
    while True:
        frame = frame_queue.get()

        # Skip some frames
        if frame_count % frame_skip == 0:
            # Resize the frame to speed up processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Detect faces
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up the face location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                color = (0, 0, 255)  # Red color for unknown face

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    color = (0, 255, 0)  # Green color for matched face

                # Draw a larger box around the face
                cv2.rectangle(frame, (left - box_offset, top - box_offset), (right + box_offset, bottom + box_offset), color, 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left - box_offset, bottom + box_offset - 35), (right + box_offset, bottom + box_offset), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left - box_offset + 6, bottom + box_offset - 6), font, 0.75, (0, 0, 0), 1)

        frame_count += 1
        processed_frames_queue.put(frame)

def display_frames():
    while True:
        frame = processed_frames_queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    read_thread = Thread(target=read_frames, args=(camera,))
    read_thread.start()

    process_thread = Thread(target=process_frames)
    process_thread.start()

    return Response(display_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #socketio.run(app, host='127.0.0.1', port=os.environ.get('PORT', 5000))
    #socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    socketio.run(app, host='localhost', port=5000, debug=True)
