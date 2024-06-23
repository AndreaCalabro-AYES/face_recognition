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
# Open the input movie file
# input_movie = cv2.VideoCapture("hamilton_clip.mp4")
frames = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fps = input_movie.get(cv2.CAP_PROP_FPS) # get the FPS
width = int(input_movie.get(3))
height = int(input_movie.get(4))


db_path = '../mm_application/encodings.json'
db = load_db(db_path)
names = [encoding['name'] for encoding in db]
known_encodings = [np.array(encoding['encoding']) for encoding in db] 

print("Encodings have been loaded", flush=True)

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True
frame_nb = 1
prev_faces_nb = 0

print("Ready to start recognition", flush=True)


while True:

    
    face_names = []    
    
    # Only process every other frame of video to save time
    if process_this_frame:
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!", flush=True)
            break
        
        # Add the delay - based on the fps of the stream
        time.sleep(1/fps)
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        cnn_face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        normal_face_locations = face_recognition.face_locations(rgb_small_frame)
        
        if len(face_locations) == 0:
            
            print("No faces are being detected")
        
        elif len(face_locations) == prev_faces_nb:
            
            print("Still here?")
        
        if len(face_locations) > prev_faces_nb:
            
            print("I see someone new")
                
            
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            # print(face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                indexes = find_true_indices(matches)
                [face_names.append(names[index]) for index in indexes]
        
            print(face_names)

        prev_faces_nb = len(face_locations)

    # Process one frame each 2 
    process_this_frame = not process_this_frame