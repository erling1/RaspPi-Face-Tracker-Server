from flask import Flask, Response
from picamera2 import Picamera2
import io

app = Flask(__name__)

picam2 = Picamera2()
video_config = picam2.create_video_configuration({"size": (640, 480)})
picam2.configure(video_config)

def generate_frames():
    picam2.start()
    stream = io.BytesIO()
    try:
        while True:
            picam2.capture_file(stream, format='jpeg')
            stream.seek(0)
            frame = stream.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            stream.seek(0)
            stream.truncate()
    finally:
        picam2.stop()

@app.route('/camera')
def camera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
