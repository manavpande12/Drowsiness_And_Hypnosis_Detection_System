from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
closed_eyes_timer = 0  
open_eyes_timer = 0 
illusion_sound_playing = False 
sleep_sound_playing = False  

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Specify your preferred resolution and framerate here
resolution = (640,480 )
framerate = 60

# Initialize the camera with specified resolution and framerate
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
cap.set(cv2.CAP_PROP_FPS, framerate)

flag = 0

detect = dlib.get_frontal_face_detector()
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predict = dlib.shape_predictor(predictor_path)

mixer.init()
sleep_sound = mixer.Sound("sleep.wav")
illusion_sound = mixer.Sound("illusion.wav")

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        if ear >= thresh:
            flag = 0
            open_eyes_timer += 1
            closed_eyes_timer = 0
        else:
            flag = 0
            closed_eyes_timer += 1
            open_eyes_timer = 0
    
    # Decision for "illusion" based on open eyes
    if open_eyes_timer > 5 * frame_check and not illusion_sound_playing:
        illusion_sound.play(-1) 
        illusion_sound_playing = True
    elif open_eyes_timer <= 5 * frame_check:
        illusion_sound.stop()
        illusion_sound_playing = False
    
    # Decision for "sleep" based on closed eyes
    if closed_eyes_timer > 2 * frame_check and not sleep_sound_playing:
        sleep_sound.play(-1)
        sleep_sound_playing = True
    elif closed_eyes_timer <= 2 * frame_check:
        sleep_sound.stop()  
        sleep_sound_playing = False
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

illusion_sound.stop()
sleep_sound.stop()
cv2.destroyAllWindows()
cap.release()
