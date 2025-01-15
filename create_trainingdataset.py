import cv2
import os
from picamera2 import Picamera2
import io



def take_pictures():

    name = input('write your name: ')
    number_id = input('Write an Integer ID: ')
    id = name +'_' + number_id

    id_person_map = {name:number_id}

    # Create a folder to save images if it doesn't exist
    folder_path = "dataset"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Constants
    COUNT_LIMIT = 30
    POS = (30, 60)  # top-left position for text
    FONT = cv2.FONT_HERSHEY_COMPLEX  # font type for text overlay
    HEIGHT = 1.5  # font scale
    TEXTCOLOR = (0, 0, 255)  # BGR - Red color for text
    BOXCOLOR = (255, 0, 255)  # BGR - Blue color for bounding box
    WEIGHT = 3  # font thickness for text
    FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Check if face detector is loaded
    if FACE_DETECTOR.empty():
        print("Error: Face detector XML file not found!")
        exit()



    picam2 = Picamera2()
    video_config = picam2.create_video_configuration({"size": (640, 480)})
    picam2.configure(video_config)
    picam2.start()
    stream = io.BytesIO()

    try:


        count = 0
        while True:
            picam2.capture_file(stream, format='jpeg')
            stream.seek(0)
            frame = stream.read()

            # Display the face detection count on the frame
            cv2.putText(frame, 'Count: ' + str(int(count)), POS, FONT, HEIGHT, TEXTCOLOR, WEIGHT)

            # Convert frame to grayscale for face detection
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = FACE_DETECTOR.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw bounding box around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), BOXCOLOR, 3)

                if cv2.waitKey(1) & 0xFF == ord(' '):

                    # Increment the count for each detected face
                    count += 1

                    # Save the cropped face as a grayscale image
                    image_filename = os.path.join(folder_path, f'{id}_image_{count}.jpg')
                    cv2.imwrite(image_filename, frameGray[y:y + h, x:x + w])

                # Optional: Stop after saving a certain number of images
                if count >= COUNT_LIMIT or cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"Saved {COUNT_LIMIT} images. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            # Show the frame with bounding boxes and count
            cv2.imshow('Face Detection', frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()

    finally:
        print(f"Picamera stopped. Images saved in {folder_path}")
        picam2.stop()



   


# Initialize the webcam
cap = cv2.VideoCapture(0)
name = input('write your name: ')
number_id = input('Write an Integer ID: ')
id = name +'_' + number_id

id_person_map = {name:number_id}

# Constants
COUNT_LIMIT = 30
POS = (30, 60)  # top-left position for text
FONT = cv2.FONT_HERSHEY_COMPLEX  # font type for text overlay
HEIGHT = 1.5  # font scale
TEXTCOLOR = (0, 0, 255)  # BGR - Red color for text
BOXCOLOR = (255, 0, 255)  # BGR - Blue color for bounding box
WEIGHT = 3  # font thickness for text
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if face detector is loaded
if FACE_DETECTOR.empty():
    print("Error: Face detector XML file not found!")
    exit()

count = 0

# Create a folder to save images if it doesn't exist
folder_path = "dataset"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

while True:
    res, frame = cap.read()

    if not res:
        print("Failed to grab frame!")
        break

    # Display the face detection count on the frame
    cv2.putText(frame, 'Count: ' + str(int(count)), POS, FONT, HEIGHT, TEXTCOLOR, WEIGHT)

    # Convert frame to grayscale for face detection
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = FACE_DETECTOR.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw bounding box around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOXCOLOR, 3)

        if cv2.waitKey(1) & 0xFF == ord(' '):

            # Increment the count for each detected face
            count += 1

            # Save the cropped face as a grayscale image
            image_filename = os.path.join(folder_path, f'{id}_image_{count}.jpg')
            cv2.imwrite(image_filename, frameGray[y:y + h, x:x + w])

        # Optional: Stop after saving a certain number of images
        if count >= COUNT_LIMIT or cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Saved {COUNT_LIMIT} images. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            break

    # Show the frame with bounding boxes and count
    cv2.imshow('Face Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()