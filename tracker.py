import cv2
import numpy as np
import os
from reciving_feed import generating_frames_array
import requests


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
    

    def train_model(self, faces, labels, path):

        self.recognizer.train(faces, np.array(labels))
        # Save the trained model to a file
        self.recognizer.save(f'{path}.xml')

        """# Create the 'trainer' folder if it doesn't exist
        if not os.path.exists("trainer"):
            os.makedirs("trainer")
        # Save the model into 'trainer/trainer.yml'
        self.recognizer.write('trainer/trainer.yml')"""


    

    def generating_frames_array(self, response):
        bytes_stream = b''  # Initialize an empty byte stream

        for chunk in response.iter_content(chunk_size=1024):
            bytes_stream += chunk  # Add the new chunk to the byte stream
            a = bytes_stream.find(b'\xff\xd8')  # Start of JPEG frame
            b = bytes_stream.find(b'\xff\xd9')  # End of JPEG frame
            
            if a != -1 and b != -1:
                jpg = bytes_stream[a:b+2]  # Extract the JPEG frame
                bytes_stream = bytes_stream[b+2:]  # Remove the processed frame from the buffer
                
                # Decode JPEG bytes to an OpenCV image
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    yield frame  # Yield the frame instead of returning
                    

        

    def facial_recognition(self):

        url = "http://192.168.1.172:8000/camera"
        
        response = requests.get(url, stream=True)

        haar_cascade = self.face_detector
        recognizer = self.recognizer
        recognizer.read('trainer/trainer.yml')
 

        # Call the generator function
        for frame in self.generating_frames_array(response):
            if frame is not None:
                cv2.imshow('Video Stream', frame)  #Remove this       Display the current frame 

                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9)

                # Iterating through rectangles of detected faces 
                for (x, y, w, h) in faces_rect: 
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                
                #have to remove imshow, and just stream the final frame flow from the server
                cv2.imshow('Detected faces', frame) 
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Cleanup resources
        cv2.destroyAllWindows()


detector = FaceDetector

faces, ids = detector.images_and_labels()

detector.train_model(faces,ids)

