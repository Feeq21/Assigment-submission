import cv2 as cv
import numpy as np
import face_recognition


capture = cv.VideoCapture(0)

# check if connected
#filePath = "../essential/forProject1.avi"
#capture = cv.VideoCapture(filePath)

if capture.isOpened() is False:
    print("Error opening camera 0")
    exit()

while capture.isOpened():
    # capture frames, if read correctly ret is True
    ret, frame = capture.read()
    
    if not ret:
        print("Didn't receive frame. Stop ")
        break
        
    facedetect = frame.copy()  
    img5 = frame.copy()
    # BRG to RGB
    rgb = frame[:, :, ::-1]
    
    # image dimension
    h, w =  facedetect.shape[:2]

    # load model
    model = cv.dnn.readNetFromTensorflow("../samples/data/opencv_face_detector_uint8.pb", "../samples/data/opencv_face_detector.pbtxt")

    # preprocessing
    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage( facedetect, 1.0, (300, 300), [
                                104., 117., 123.], False, False)

    # set blob asinput and detect face
    model.setInput(blob)
    detections = model.forward()

    faceCounter = 0
    # draw detections above limit confidence > 0.7
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        #
        if confidence > 0.7:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle( facedetect, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = "{:.3f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText( facedetect, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Detect 5 landmarks:
    face_landmarks_list_5 = face_recognition.face_landmarks(rgb, None, "small")

    #print(face_landmarks_list_5)
    # Draw all detected landmarks:
    for face_landmarks in face_landmarks_list_5:
        for facial_feature in face_landmarks.keys():
            for p in face_landmarks[facial_feature]:
                cv.circle(facedetect, p, 2, (0, 0, 255), -1)

    # display frame
    cv.imshow("Face detection",  facedetect)
    #cv.imshow("5 Landmarks",  img5) 
   
    k = cv.waitKey(1) 
    # check if key is q then exit
    if k == ord("q"):
        break

capture.release()
cv.destroyAllWindows()