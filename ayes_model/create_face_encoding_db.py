"""
This file is intended to create the database used to recognize the faces for the ayes application
1. Add your pic to the train_files folder, naming it as you want to be recognized (we will check if it exists already in the encodings file, please don't touch that)
2. Add pic/videos to the test_files folder, we will use it to check the encoding (try to be naughty)
3. 

"""

import os
import json
import face_recognition
import numpy as np

def load_encodings(encodings_path):
    """
    Load the existing encodings json file, to get the existing encodings 
    """
    if os.path.exists(encodings_path):
        with open(encodings_path, 'r') as file:
            try: 
                encodings = json.load(file)
            except json.decoder.JSONDecodeError:
                encodings = []
                
    else:
        encodings = []
    
    return encodings



def save_encodings(encodings_path, encodings):
    """
    Save the new encodings in the encodings.json file
    """
    with open(encodings_path, 'w') as file:
        json.dump(encodings, file, indent=4)

def create_encoding(image_path):
    """
    Use the face recognition api to load the image and create the new encoding
    """
    picture = face_recognition.load_image_file(file=image_path)
    encoding = face_recognition.face_encodings(picture, num_jitters=50, model="large")[0]
    return encoding

def test_function(encoding, name, test_dir):
    """
    Here we test only against pictures, for the video use the specific path
    """
    encoding = np.array(encoding)
    for test_filename in os.listdir(test_dir):
        test_name, test_ext = os.path.splitext(test_filename)
        if (name in test_name) and (test_ext.lower() in ['.jpg', '.jpeg', '.png']):
            print(f"Testing if {name} is recognized in {test_filename}")
            test_face_encoding = create_encoding(os.path.join(test_dir, test_filename))
            result = face_recognition.compare_faces([encoding], test_face_encoding)
    return result

def process_train_files(train_dir, encodings):
    """
    Look for the existing names in the 
    """
    existing_names = {entry['name'] for entry in encodings}
    new_encodings = []
    
    for filename in os.listdir(train_dir):
        name, ext = os.path.splitext(filename)
        if name not in existing_names and ext.lower() in ['.jpg', '.jpeg', '.png']:
            image_path = os.path.join(train_dir, filename)
            new_encoding = create_encoding(image_path)
            print(f"New encoding created {name}")
             
            new_encodings.append({'name': name, 'encoding': new_encoding.tolist()})
    
    return new_encodings

def main():
    encodings_path = '../mm_application/encodings.json'
    train_dir = './train_files'
    test_dir = './test_files'
    
    encodings = load_encodings(encodings_path)
    
    new_encodings = process_train_files(train_dir, encodings)
    
    for new_encoding in new_encodings:
        name = new_encoding['name']
        encoding = new_encoding['encoding']
        
        if test_function(encoding, name, test_dir):
            encodings.append(new_encoding)
        else:
            print(f"Encoding for {name} failed the test")
    
    save_encodings(encodings_path, encodings)

if __name__ == "__main__":
    main()