"""
Here we want to test the real scenario of the webcam stream on the raspberry py. 
We will also add small delays to have a complete view of the process. Further analysis will improve the model. 
"""
import os
import json
import face_recognition
import numpy as np
import cv2
import time
from mqtt_client_ayes import AyesMqttClient



def load_db(db_path):
    """
    Load the existing encodings json file, to get the existing encodings 
    """
    if os.path.exists(db_path):
        with open(db_path, 'r') as file:
            try: 
                db = json.load(file)
            except json.decoder.JSONDecodeError:
                db = []
                
    else:
        db = []
    
    # for encoding in encodings:
    #     encoding = np.array(encoding)
    
    return db

def load_test_video(test_dir, filename):
    for test_filename in os.listdir(test_dir):
        test_name, test_ext = os.path.splitext(test_filename)
        if test_name == filename:
            video_path = os.path.join(test_dir, test_filename)
    
    return cv2.VideoCapture(video_path)


def find_true_indices(boolean_list):
    """
    Find the indexes of the recognized persons 
    """
    return [index for index, value in enumerate(boolean_list) if value]
   

 
test_dir = './test_files'  
input_movie = load_test_video(test_dir, "Video Test")
frames = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fps = input_movie.get(cv2.CAP_PROP_FPS) # get the FPS
width = int(input_movie.get(3))
height = int(input_movie.get(4))


db_path = './encodings.json'
db = load_db(db_path)
names = [encoding['name'] for encoding in db]
known_encodings = [np.array(encoding['encoding']) for encoding in db] 

print("Encodings have been loaded")
print(names)
MQTT_TOPICS = ["greetings/face_added",
               "greetings/face_removed"]

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60
CLIENT_ID = "FaceRecognition"
MQTT_BROKER_HOST = "mqtt_broker"
MQTT_BROKER_PORT = 1883

face_recognition_client = AyesMqttClient(
    broker= MQTT_BROKER_HOST,
    port= MQTT_BROKER_PORT,
    topics_list= MQTT_TOPICS,
    client_id= CLIENT_ID
)


# Initialize some variables
face_locations = []
face_encodings = []
sent_faces = []
frame_nb = 1
prev_faces_nb = 0
count = 0
retry_recognition = False # Flag to retry if we got a new encoding, and such encoding is unknown 
publish_flag = False # Flag to publish on mqtt
retry_next_frame = False
new_unknown_detected = False

FRAMES_JUMP = 10

print("Ready to start recognition")

time.sleep(2)

face_recognition_client.connect()

while True:

    # Initialization
    face_added_names = []   
    face_removed_names = []
    count += 1 

    # Only process every other FRAMES_JUMP of video to save time
    if count % FRAMES_JUMP == 0:
        print(f"second: {count/fps}")
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!")
            check = False
            break
        
        # Add the delay - based on the fps of the stream
        time.sleep(1/fps)
        
        # Resize frame of video 70% size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        # print(len(face_locations))
        if len(face_locations) == 0:
            
            print("No faces are being detected")
            publish_flag = False
            
        
        elif (len(face_locations) == prev_faces_nb) and (not retry_next_frame):
            
            print("Still here?")
            publish_flag = False
            
        
        elif (len(face_locations) > prev_faces_nb) or retry_next_frame:
            
            publish_flag = True
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            if not retry_next_frame:
            
                print("I see someone new")
                
                # print(face_locations)

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    indexes = find_true_indices(matches)
                    
                if indexes:
                    # Recognize matched faces
                    for index in indexes:
                        face_added_names.append(names[index])
                else:
                    face_added_names.append("Unknown")
                    new_unknown_detected = True  # Mark that an unknown face was detected
                
                if new_unknown_detected:
                    print("There is an unknown person")
                    publish_flag = False
                    retry_next_frame = True  # Set flag to retry in the next processed frame
            
            elif retry_next_frame:
                print("I will retry to recognize again such unknown")
                retry_next_frame = False
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    indexes = find_true_indices(matches)
                if len(indexes) == len(face_encodings):
                    for index in indexes:
                        face_added_names.append(names[index])
                        face_removed_names.append("Unknown")
                        
                else:
                    publish_flag = True
                    # face_names.append("Unknown")
                    face_added_names.append(names[index])
                    face_added_names.append("Unknown")
                    print("You are not in my system")
                    new_unknown_detected = False  # Mark that we retried
                
    
        
        # Publish only when necessary
        if publish_flag:
            face_added_names = list(set(face_added_names))
            face_added = json.dumps({"names" : face_added_names})
            face_recognition_client.publish_message("greetings/face_added", face_added)
            face_removed = json.dumps({"names" : face_removed_names})
            face_recognition_client.publish_message("greetings/face_removed", face_removed)
            

        prev_faces_nb = len(face_locations)
    else:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!")
            check = False
            break
        
    
