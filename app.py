from flask import Flask, render_template, Response
import cv2
import threading

app = Flask(__name__)

# Global variables for video capture and sign language detection
cap = None
is_thread_running = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_video_capture():
    global cap, is_thread_running
    cap = cv2.VideoCapture(0)
    while True:
        if is_thread_running:
            ret, frame = cap.read()
            # Your sign language detection code goes here
            # You may need to modify the code to process 'frame' and update the results accordingly
            # For simplicity, you can keep the existing detection code in the main() function
        else:
            break
    cap.release()

@app.route('/start_detection')
def start_detection():
    global is_thread_running
    if not is_thread_running:
        is_thread_running = True
        detection_thread = threading.Thread(target=start_video_capture)
        detection_thread.start()
    return 'Detection started'

@app.route('/stop_detection')
def stop_detection():
    global is_thread_running
    is_thread_running = False
    return 'Detection stopped'

if __name__ == '__main__':
    app.run(debug=True)
