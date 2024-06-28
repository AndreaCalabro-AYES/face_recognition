# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.



import face_recognition
import numpy as np
import cv2
import os
import json

# Load the database of encodings
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
    
    return db

def get_knwon_info(db_path = "../mm_application/encodings.json"):
    """
    Load the db from the path, and extract
        1. the names usign the "name" kwd of the json file 
        2. the encodings using the "encodings" kwd of the json file, 
           and transofrm them in a np.array to be used by the face_recognition
    """
    db = load_db(db_path)
    names = [encoding['name'] for encoding in db]
    known_encodings = [np.array(encoding['encoding']) for encoding in db] 
    
    return names, known_encodings


# 
def find_true_indices(boolean_list):
    """
    Find the indexes of the recognized persons 
    """
    return [index for index, value in enumerate(boolean_list) if value]



# KEPT LIKE THIS AS THIS MAY MOVE AROUND IN THE FUTURE
DB_PATH = "../mm_application/encodings.json"

print("Loading known people information", flush=True)

names, known_encodings = get_knwon_info(db_path=DB_PATH)

print("Encodings have been loaded", flush=True)



cap = cv2.VideoCapture(0)

def main():
    
    # Initialize variables
    # Used to define how often we capture the image
    FRAMES_JUMP = 10
    face_locations = []
    face_encodings = []
    count = 0
    prev_faces_nb = 0
    
    
    if not cap.isOpened():
        print("Can't open webcam")
        return 1
        

    while True:
        
        
        face_names = []   
        count += 1 
        
        # Only process every other FRAMES_JUMP of video to save time
        if count % FRAMES_JUMP == 0:
            
            ret, frame = cap.read()
            
            if not ret:
                print("Ciao")
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            # face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if len(face_locations) == 0:
                
                print("No faces are being detected", flush=True)
            
            elif len(face_locations) == prev_faces_nb:
                
                print("Still here?", flush=True)
            
            elif len(face_locations) > prev_faces_nb:
                
                print("I see someone new", flush=True)
                    
                
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    indexes = find_true_indices(matches)
                    [face_names.append(names[index]) for index in indexes]
            
                print(face_names)

            prev_faces_nb = len(face_locations)
            
        else:
            ret, frame = cap.grab()
            if not ret:
                print("Finished")
                break
        
        
if __name__ == "__main__":
    main()
