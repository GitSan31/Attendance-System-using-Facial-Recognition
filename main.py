
# pip install cmake
# pip install face_recognition
# pip install opencv-python
# pip install numpy

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import winsound


video_capture = cv2.VideoCapture(0)


# load known faces


harry_image = face_recognition.load_image_file("faces/harry.jpg")
harry_encoding = face_recognition.face_encodings(harry_image)[0]
selmon_image = face_recognition.load_image_file("faces/selmon.jpg")
selmon_encoding = face_recognition.face_encodings(selmon_image)[0]
bachan_image = face_recognition.load_image_file("faces/bachan.jpg")
bachan_encoding = face_recognition.face_encodings(bachan_image)[0]
twinkle_image = face_recognition.load_image_file("faces/twinkle.jpg")
twinkle_encoding = face_recognition.face_encodings(twinkle_image)[0]



known_face_encodings = [harry_encoding,selmon_encoding,bachan_encoding,twinkle_encoding]
known_face_names =["Harry", "Salman", "Bachan","Twinkle"]


# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# get current date and time

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)


#function to play sound for face detection
def play_sound():
    duration = 1000 #milliseconds
    frequency = 1000 #Hz
    winsound.Beep(frequency,duration)
while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    name="Unknown"
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]


        # Add the text if the person is present
        if name!="Unknown":

            if name in known_face_names:
                font=cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (175, 400)
                fontScale = 1.5
                fontColor = (0, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + "Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,lineType)

                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

                    #play sound for face detection
                    play_sound()


    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

