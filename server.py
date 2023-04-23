#pip3 install flask

import cv2
from flask import Flask, Response

app = Flask(__name__)

cap = None

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print(' Got no frame ')
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def hello():
    return 'Hello, World!'

print('Starting server')

if __name__ == '__main__':
    if cap is not None :
        cap.release
    print('getting cv2.VideoCapture(0)')
    cap = cv2.VideoCapture(0)    
    print('got cv2.VideoCapture(0)')

    app.run(host='0.0.0.0', port=5000, debug=True)

