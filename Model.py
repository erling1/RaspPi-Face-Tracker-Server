import cv2
import numpy as np
import os
import requests
from typing import Iterator

class FaceDetector():

    def __init__(self):
        #Creating an instance of LBPHFaceRecognizer, which will be trained on our custom dataset.
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        #Creating an instance of our face detector 
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



    def images_and_labels(self, folder_path):
        training_images = []
        ids = []
        faces = []
        for filename in os.listdir(folder_path):
            #reading Image
            img = cv2.imread(os.path.join(folder_path,filename),cv2.IMREAD_GRAYSCALE)
            rectangle_faces_coords = self.face_detector.detectMultiScale(img)

            #getting name of the person in the picture
            words = filename.split('_')
            ID_person = int(words[-1])

            ids.append(ID_person)
            training_images.append(rectangle_faces_coords)

            

            for (x,y,w,h) in rectangle_faces_coords:
                faces.append(img[y:y+h, x:x+w])

        return faces,ids  
    

    def train_model(self, faces, labels, file_path):

        self.recognizer.train(faces, np.array(labels))
        # Save the trained model to a file
        self.recognizer.save(file_path)

        """# Create the 'trainer' folder if it doesn't exist
        if not os.path.exists("trainer"):
            os.makedirs("trainer")
        # Save the model into 'trainer/trainer.yml'
        self.recognizer.write('trainer/trainer.yml')"""

        

    def facial_recognition_tracker(self, camera_feed: Iterator[np.NDArray]):

        haar_cascade = self.face_detector
        recognizer = self.recognizer
        recognizer.read('trainer/trainer.yml')
 

        # Call the generator function....  camera feed > respons
        for frame in camera_feed:

            numpy_array = np.frombuffer(frame, dtype=np.uint8)
            # Decode JPEG bytes to OpenCV image
            cv2_frame = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

            if cv2_frame is not None:
                cv2.imshow('Video Stream', cv2_frame)  #Remove this       Display the current frame 

                gray_img = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY) 
                faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9)

                rects = []
                # Iterating through rectangles of detected faces 
                for (x, y, w, h) in faces_rect:
                    rects.append((x,y,w,h)) 
                    cv2.rectangle(cv2_frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 


                yield (cv2_frame, rects)
                


detector = FaceDetector

faces, ids = detector.images_and_labels()

detector.train_model(faces,ids)

