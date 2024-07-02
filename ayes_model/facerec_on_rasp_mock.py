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
import paho.mqtt.client as mqtt



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


db_path = '../mm_application/encodings.json'
db = load_db(db_path)
names = [encoding['name'] for encoding in db]
known_encodings = [np.array(encoding['encoding']) for encoding in db] 

print("Encodings have been loaded", flush=True)
MQTT_TOPICS = ["greetings/face_added",
               "greetings/face_removed"]

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60
CLIENT_ID = "FaceRecognition"
MQTT_BROKER_HOST = "mqtt_broker"
MQTT_BROKER_PORT = 1883

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print(f"Connected to broker at {MQTT_BROKER_HOST}: {MQTT_BROKER_PORT}", flush=True)
        for mqtt_topic in MQTT_TOPICS:
            client.subscribe(mqtt_topic)
    else:
        print("Failed to connect, return code %d", rc, flush=True)

def on_subscribe(client, userdata, mid, reason_code_list, properties):
    if reason_code_list[0].is_failure:
        print(f"Broker rejected you subscription: {reason_code_list[0]}", flush=True)
    else:
        print(f"Broker granted the following QoS: {reason_code_list[0].value}", flush=True)

def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()} on topic {msg.topic}", flush=True)

def on_disconnect(client, userdata, rc):
    print("Disconnected with result code: %s", rc, flush=True)
    reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
    while reconnect_count < MAX_RECONNECT_COUNT:
        print("Reconnecting in %d seconds...", reconnect_delay, flush=True)
        time.sleep(reconnect_delay)

        try:
            client.reconnect()
            print("Reconnected successfully!", flush=True)
            return
        except Exception as err:
            print("%s. Reconnect failed. Retrying...", err, flush=True)

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1
    print("Reconnect failed after %s attempts. Exiting...", reconnect_count, flush=True)

# Set Connecting Client ID
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)

# client.username_pw_set(username, password)
mqtt_client.on_connect = on_connect
mqtt_client.on_subscribe = on_subscribe
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

def connect():
    mqtt_client.connect(
        host = MQTT_BROKER_HOST,
        port = MQTT_BROKER_PORT,
        keepalive = 60
    )
    print(f"Connected to broker at {MQTT_BROKER_HOST}: {MQTT_BROKER_PORT}", flush=True)

    mqtt_client.loop_start()

def publish(topic, message):
    result = mqtt_client.publish(topic, message)
    status = result[0]
    if status == 0:
        print(f"Send `{message}` to topic `{topic}`", flush=True)
    else:
        print(f"Failed to send message to topic {topic}", flush=True)

connect()


# Initialize some variables
face_locations = []
face_encodings = []
frame_nb = 1
prev_faces_nb = 0

FRAMES_JUMP = 10

print("Ready to start recognition", flush=True)

time.sleep(30)

count = 0

while True:

    
    face_names = []   
    count += 1 

    # Only process every other FRAMES_JUMP of video to save time
    if count % FRAMES_JUMP == 0:
        
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!", flush=True)
            check = False
            break
        
        # Add the delay - based on the fps of the stream
        time.sleep(1/fps)
        
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
            # print(face_locations)

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                indexes = find_true_indices(matches)
                [face_names.append(names[index]) for index in indexes]
        
            print(face_names)
            data = json.dumps({"names" : face_names})
            mqtt_client.publish("greetings/face_added", data)

        prev_faces_nb = len(face_locations)
    else:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_nb += 1
        if frame_nb > frames:
            print("finished!", flush=True)
            check = False
            break
        
    
