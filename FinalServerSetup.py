import RPi.GPIO as GPIO
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
from pid import PID

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
     

      
servoRange = (-90, 90)

#@app.route("/track")
def track_face( objX, objY, centerX, centerY):
     
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
        ((objX.value, objY.value), rect) = object_Lock

        if rect is not None:
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),2)
            
        # display the frame to the screen
        cv2.imshow("Pan-Tilt Face Tracking", frame)
        cv2.waitKey(1)
	     
def signal_handler():
     # print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")
    # disable the servos
    pth.servo_enable(1, False)
    pth.servo_enable(2, False)
    # exit
    sys.exit()
     

def pid_process(output, p, i, d, objCoord, centerCoord):
	# signal trap to handle keyboard interrupt
	signal.signal(signal.SIGINT, signal_handler)
	# create a PID and initialize it
	p = PID(p.value, i.value, d.value)
	p.initialize()
	# loop indefinitely
	while True:
		# calculate the error
		error = centerCoord.value - objCoord.value
		# update the value
		output.value = p.update(error)



def in_range(val, start, end):
	# determine the input value is in the supplied range
	return (val >= start and val <= end)

def set_servos(pan, tlt):
	# signal trap to handle keyboard interrupt
	signal.signal(signal.SIGINT, signal_handler)
	# loop indefinitely
	while True:
		# the pan and tilt angles are reversed
		panAngle = -1 * pan.value
		tiltAngle = -1 * tlt.value
		# if the pan angle is within the range, pan
		if in_range(panAngle, servoRange[0], servoRange[1]):
			pth.pan(panAngle)
		# if the tilt angle is within the range, tilt
		if in_range(tiltAngle, servoRange[0], servoRange[1]):
			pth.tilt(tiltAngle)
          
          
               

"""

@app.route('/camera')
def camera():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
"""


if __name__ == '__main__':
   
   with Manager() as manager:
    # enable the servos
    pth.servo_enable(1, True)
    pth.servo_enable(2, True)
    
    # set integer values for the object center (x, y)-coordinates
    centerX = manager.Value("i", 0)
    centerY = manager.Value("i", 0)
    # set integer values for the object's (x, y)-coordinates
    objX = manager.Value("i", 0)
    objY = manager.Value("i", 0)
    # pan and tilt values will be managed by independed PIDs
    pan = manager.Value("i", 0)
    tlt = manager.Value("i", 0)

    # set PID values for panning
    panP = manager.Value("f", 0.09)
    panI = manager.Value("f", 0.08)
    panD = manager.Value("f", 0.002)
    # set PID values for tilting
    tiltP = manager.Value("f", 0.11)
    tiltI = manager.Value("f", 0.10)
    tiltD = manager.Value("f", 0.002)


    processObjectCenter = Process(target=track_face,args=(objX, objY, centerX, centerY))
	processPanning = Process(target=pid_process,args=(pan, panP, panI, panD, objX, centerX))
	processTilting = Process(target=pid_process,args=(tlt, tiltP, tiltI, tiltD, objY, centerY))
	processSetServos = Process(target=set_servos, args=(pan, tlt))
    

