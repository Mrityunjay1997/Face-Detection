import cv2
import dlib


video=cv2.VideoCapture(0)                             # for inbuilt webcam

detector=dlib.get_frontal_face_detector()             #detect front face from webcam

while True:                                            # while infinite loop to capture image to video
    frame: object
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # convert color image to grey for accurate detection
    faces = detector(gray)

    num = 0                                          # stores no. of faces , initially it is zero
    for face in faces:
        x, y = face.left(), face.top()
        hi, wi = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (hi, wi), (0, 0, 255), 2)   # face will be shown in rectangle frame
        num = num + 1

        cv2.putText(frame, 'face' + str(num), (x - 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('faces', frame)

    if cv2.waitKey(1) == ord('q'):
       break

video.release()
cv2.destroyAllWindows()
