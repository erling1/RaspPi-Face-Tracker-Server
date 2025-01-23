
from picamera2 import Picamera2
import io

import os
import numpy as np
import cv2
import glob 
from flask import Flask, flash, request, redirect, url_for,Response
from werkzeug.utils import secure_filename
from multiprocessing import Manager
from multiprocessing import Process
from imutils.video import VideoStream

from Model import FaceDetector
from objectcenter import ObjCenter

import sys
import signal
import time

app = Flask(__name__)


app = Flask(__name__)
UPLOAD_FOLDER = 'trainingimages'  
YML_PATH = 'trainer'
ALLOWED_EXTENSIONS = { 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_frames_RaspPi():
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration({"size": (640, 480)})
    picam2.configure(video_config)
    picam2.start()
    stream = io.BytesIO()
    try:
        while True:
            picam2.capture_file(stream, format='jpeg')
            stream.seek(0)
            frame = stream.read() #JPEG encoded data 
            yield frame
            stream.seek(0)
            stream.truncate()
    finally:
        picam2.stop()

        

@app.route('/uploadpictures', methods=['POST'])
def upload_images():

    if 'images' not in request.files:
            flash('No file part')
            return redirect(request.url)
    
    image = request.files['images']

    safe_filename = secure_filename(image.filename)

    image.save(os.path.join(app.config['UPLOAD_FOLDER'], safe_filename))


def get_images_from_Flask_Server():
     
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    pattern = os.path.join(UPLOAD_FOLDER, '*.jpeg')
 
    image_paths = glob.glob(pattern)

    return image_paths
     

def activate_model():
     
     model = FaceDetector

     image_paths = get_images_from_Flask_Server()

     faces,labels =  model.images_and_labels(image_paths)


     trainer_file_path = os.path.join(YML_PATH,'trainer.yml')

     model.train_model(faces=faces,labels=labels,path=trainer_file_path)

     feed = generate_frames_RaspPi() #Creats iterator object to be used in a for loop

     model.facial_recognition_tracker(feed)

      


@app.route("/track")
def track_face():
     
     image_paths = get_images_from_Flask_Server()

     model = FaceDetector

     faces,labels =  model.images_and_labels(image_paths)

     trainer_file_path = os.path.join(YML_PATH,'trainer.yml')

     model.train_model(faces=faces,labels=labels,path=trainer_file_path)

     feed = generate_frames_RaspPi() #Creats iterator object to be used in a for loop

     for frame, rects in model.facial_recognition_tracker(feed):

        (H, W) = frame.shape[:2]

        centerX.value =  W // 2
        centerY.value = H //2 

        object_Lock = ObjCenter(rects, (centerX, centerY))
        ((objX.value, objY.value), rect) = objectLock

        if rect is not None:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
				2)
            
        # display the frame to the screen
		cv2.imshow("Pan-Tilt Face Tracking", frame)
		cv2.waitKey(1)






          
          ObjCenter
          
          
               



@app.route('/camera')
def camera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)