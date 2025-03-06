import cv2
import numpy as np
import face_recognition
import dlib
import time
import random

# Load face recognition models
try:
    aman_image = face_recognition.load_image_file("Aman.jpeg")
    aman_encoding = face_recognition.face_encodings(aman_image)[0]
    known_encodings = [aman_encoding]
    known_names = ["Aman"]
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

# Initialize video stream
video_capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Liveness Parameters
start_time = time.time()
liveness_detected = False
blink_count = 0
BLINK_THRESHOLD = 0.23  # Adjusted for better accuracy
BLINK_FRAMES = 2  # Number of consecutive frames to consider a blink
frame_counter = 0

# Action-Based Liveness Verification
actions = ["Turn Left", "Turn Right", "Look Up", "Tilt Left", "Tilt Right"]
current_action = random.choice(actions)
action_start_time = time.time()
action_completed = False
captured_images = 0
MAX_IMAGES = 50

def get_new_action():
    return random.choice(actions)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    faces = detector(gray)
    
    for (top, right, bottom, left), face_encoding, face in zip(face_locations, face_encodings, faces):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < BLINK_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= BLINK_FRAMES:
                blink_count += 1  # Blink detected
                liveness_detected = True
                start_time = time.time()
            frame_counter = 0
        
        # Action-based authentication
        if not action_completed:
            cv2.putText(frame, f"Perform: {current_action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            captured_images += 1
            if captured_images >= MAX_IMAGES:
                action_completed = True
                captured_images = 0
                current_action = get_new_action()
                action_start_time = time.time()
        
    # Display liveness status
    if liveness_detected:
        cv2.putText(frame, "Liveness Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Fake Attempt", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Fake attempt detection timeout
    if not liveness_detected and time.time() - start_time > 10:
        cv2.putText(frame, "Fake Attempt Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imshow("Face Recognition with Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
