"""
In here I would like to test using the x11 server
We have as input a video that has been recorded offline,
As output the same video displayed on the x11 host with a box around the face and the name of the recognized person
"""



import os
import json
import face_recognition
import numpy as np
import cv2

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

# Create an output movie file (make sure resolution/frame rate matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID') # specify the video code
# output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# # Load some sample pictures and learn how to recognize them.
# lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
# lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

# al_image = face_recognition.load_image_file("alex-lacamoire.png")
# al_face_encoding = face_recognition.face_encodings(al_image)[0]

db_path = '../mm_application/encodings.json'
db = load_db(db_path)
names = [encoding['name'] for encoding in db]
known_encodings = [np.array(encoding['encoding']) for encoding in db] 

# print(names)
# print(encodings)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frame_nb = 1


while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_nb += 1
    if frame_nb > frames:
        print("finished!")
        break
        
    
    
    

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # print(face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            indexes = find_true_indices(matches)
            for index in indexes:
                print(names[index])

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        # name = None
        # if match[0]:
        #     name = "Lin-Manuel Miranda"
        # elif match[1]:
        #     name = "Alex Lacamoire"

        # face_names.append(name)

#     # Label the results
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         if not name:
#             continue

#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Draw a label with a name below the face
#         cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

#     # Write the resulting image to the output video file
#     print("Writing frame {} / {}".format(frame_number, length))
#     output_movie.write(frame)

# # All done!
# input_movie.release()
# cv2.destroyAllWindows()
