import cv2
import os
import time
import wave
import numpy as np

from keras.models import load_model
from simpleaudio import WaveObject as wo

face = cv2.CascadeClassifier(
    r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(
    r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(
    r'haar cascade files\haarcascade_righteye_2splits.xml')
model = load_model('models/cnncat2.h5')

sound = wo.from_wave_file('alarm.wav')
sound_obj = None

path = os.getcwd()
color_line_BGR = (0, 0, 0) # Lavender = (234, 206, 199)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
rpred, lpred = [], []

def draw_box(frame, height, end=None, color=(0, 0, 0), thickness=cv2.FILLED):
    '''
    Draw a black (default) box at the bottom left of the frame
    '''
    start = (0, height-50)
    if end is None:
        end = (320, height)
    cv2.rectangle(frame, start, end, color, thickness=thickness)

def put_text(frame, text, height):
    '''
    Put a white text on the frame which will within the black box
    '''
    cv2.putText(frame, text, (10, height-20), font,
                1, (255, 255, 255), 1, cv2.LINE_AA)

def detect_eye(eye, frame, face, x_full_img, y_full_img):

    '''
    Detect eye from the given face.
    PS: frame is needed for drawing box for detected eye

    Future Improvement:
    1) Eliminate the detection of same eye (both right eye or both left eye)
    '''
    one_side_eye = eye.detectMultiScale(face)

    for (x, y, w, h) in one_side_eye:

        # Uncomment to reveal the drawing box of eye
        # x_full = x + x_full_img
        # y_full = y + y_full_img
        # cv2.rectangle(frame, (x_full, y_full), (x_full+w, y_full+h), color_line_BGR, 1)

        eye = face[y:y+h, x:x+w]
        eye = cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY)
        eye = cv2.resize(eye, (24, 24))
        eye = eye/255
        eye = eye.reshape(24, 24, -1)
        eye = np.expand_dims(eye, axis=0)
        pred = model.predict_classes(eye)
        return pred # Read one eye only

    return []

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    faces = face.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    if len(faces) > 0:
        draw_box(frame, height)
        x, y, w, h = faces[0]   # Read 1 face only
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_line_BGR, 1)
        face_region = frame[y:y+h, x:x+h]   # Crop the face image

        # Increase the brightness of face to detect eye under dim condition.
        # By refer to method 2 of this website: 
        # https://answers.opencv.org/question/193276/how-to-change-brightness-of-an-image-increase-or-decrease/
        # Improvement:
        # 1) To apply this function when the picture is dim (Detect the brightness of image)
        face_region = cv2.add(face_region, np.array([100.0]))
        # cv2.imshow('face', face_region)
        
        # Detect eyes by checking on the cropped face
        rpred = detect_eye(leye, frame, face_region, x, y)
        lpred = detect_eye(reye, frame, face_region, x, y)

        if len(rpred) > 0 and len(lpred) > 0:
            # When both eye is detected
            if (rpred[0] == 0 and lpred[0] == 0):
                # When both eye is opened
                score += 1
            else:
                score -= 1

        if score < 10:
            if score < 0:
                score = 0
            put_text(frame, "Driver is Alert", height)
        elif score < 25:
            put_text(frame, "Ngantuk ke?", height)  # Ngantuk = sleepy in Malay Language
            
            try:
                sound_obj.stop()
            except AttributeError:
                pass
        else:
            put_text(frame, "Stop Now!", height)

            if score >= 30:
                # Maximum score
                score = 30

            # person is feeling sleepy so we beep the alarm
            try:
                sound_obj = sound.play()
            except:  # isplaying = False
                pass
            
            # Uncomment to write the image
            # cv2.imwrite(os.path.join(path, 'image.jpg'), frame)

            if(thicc < 16):
                thicc = thicc + 2
            else:
                thicc = thicc-2
                if(thicc < 2):
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            
        cv2.putText(frame, 'Score:'+str(score), (220, height-20),
                    font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        draw_box(frame, height, end=(225, height))

        cv2.putText(
            frame, 'No face detected', (10, height-20),
            font, 1, (255, 255, 255), 1, cv2.LINE_AA)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
