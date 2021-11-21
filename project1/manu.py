import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
import sys

def q1a():
    img = cv.imread(r'D:/KOTOCLASS/firstmonth/cv-master/essential/mypaintshape.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image:
    ret, threshImg = cv.threshold(imgGray, 150, 255, cv.THRESH_BINARY_INV)

    # Find contours using the thresholded image:
    contours, hierarchy = cv.findContours(threshImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # number of detected contours:
    # print(f"The total number of object: {len(contours)}")

    # create list of tuple (size, shape) for each contour
    # list of contour size
    contours_sizes = [cv.contourArea(contour) for contour in contours]
    # list of (size, contour)
    size_shape_list = zip(contours_sizes, contours)
    sorted_size_shape_list = sorted(size_shape_list)
    # (contour_sizes, contours) = zip(*sorted_size_shape_list)
    # print(imgGray[250,50])
    plt.figure(figsize=(10, 10))

    # BGR to RGB
    imgRGB = img[:, :, ::-1]
    plt.subplot(221)
    plt.title(f"The total number of object: {len(contours)}")
    plt.imshow(imgRGB)

    imgSize = img.copy()
    for i, (size, contour) in enumerate(sorted_size_shape_list):
        # Compute the moment of contour:
        M = cv.moments(contour)

        # The center or centroid can be calculated as follows:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Get the position to draw:
        text = str(i + 1)
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 3
        text_size = cv.getTextSize(text, fontFace, fontScale, thickness)[0]

        text_x = cX - text_size[0] / 2
        text_x = round(text_x)
        text_y = cY + text_size[1] / 2
        text_y = round(text_y)

        # Write the ordering of the shape on the center of shapes
        color = (0, 0, 0)
        cv.putText(imgSize, text, (text_x, text_y), fontFace, fontScale, color, thickness)

    # approxPolyDP():
    imgApproxPolyDP = img.copy()
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        epsilon = 0.03 * perimeter
        approxPolyDP = cv.approxPolyDP(contour, epsilon, True)

        color = (0, 255, 255)
        thickness = 5

        # draw line
        for approx in approxPolyDP:
            cv.drawContours(imgApproxPolyDP, [approx], 0, color, thickness)
        color = (0, 0, 255)
        thickness = 5
        # draw points
        for approx in [approxPolyDP]:
            # draw points
            squeeze = np.squeeze(approx)
            # print('contour:',approx.shape, squeeze.shape)
            for p in squeeze:
                pp = tuple(p.reshape(1, -1)[0])
                cv.circle(imgApproxPolyDP, pp, 10, color, -1)

        # determine shape
        verticeNumber = len(approxPolyDP)
        if verticeNumber == 3:
            vertice_shape = (verticeNumber, 'Triangle')
        elif verticeNumber == 4:
            # get aspect ratio
            x, y, width, height = cv.boundingRect(approxPolyDP)
            aspectRatio = float(width) / height
            # print(aspectRatio)
            if 0.90 < aspectRatio < 1.1:
                vertice_shape = (verticeNumber, 'Square')
            else:
                vertice_shape = (verticeNumber, 'Rectangle')
        elif verticeNumber == 5:
            vertice_shape = (verticeNumber, 'Pentagon')
        elif verticeNumber == 6:
            vertice_shape = (verticeNumber, 'Hexagon')
        else:
            vertice_shape = (verticeNumber, 'Circle')

        # write shape
        # Compute the moment of contour:
        M = cv.moments(contour)

        # The center or centroid can be calculated as follows:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        # Get the position to draw:

        text = vertice_shape[1]
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 3
        text_size = cv.getTextSize(text, fontFace, fontScale, thickness)[0]

        text_x = cX - text_size[0] / 2
        text_x = round(text_x)
        text_y = cY + text_size[1] / 2
        text_y = round(text_y)

        # Write the ordering of the shape on the center of shapes
        color = (0, 0, 0)
        cv.putText(imgApproxPolyDP, text, (text_x, text_y), fontFace, fontScale, color, thickness)

    # BGR to RGB
    imgRGB = imgSize[:, :, ::-1]
    plt.subplot(223)
    plt.title("Sorted by Size")
    plt.imshow(imgRGB)

    imgRGB = imgApproxPolyDP[:, :, ::-1]
    plt.subplot(224)
    plt.title("Sorted by Shape")
    plt.imshow(imgRGB)

    plt.tight_layout()
    plt.show()

def q1b():
    # Load the image and convert it to grayscale:
    img = cv.imread(r'D:/KOTOCLASS/firstmonth/cv-master/essential/mypaintshape.png')

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image:
    ret, threshImg = cv.threshold(imgGray, 150, 255, cv.THRESH_BINARY_INV)

    # Find contours using the thresholded image:
    contours, hierarchy = cv.findContours(threshImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # number of detected contours:
    #print(f"The total number of object: {len(contours)}")

    # create list of tuple (size, shape) for each contour
    # list of contour size
    contours_sizes = [cv.contourArea(contour) for contour in contours]
    # list of (size, contour)
    size_shape_list = zip(contours_sizes, contours)
    sorted_size_shape_list = sorted(size_shape_list)
    # (contour_sizes, contours) = zip(*sorted_size_shape_list)
    #print(imgGray[250,50])
    plt.figure(figsize=(10,10))

    # BGR to RGB
    imgRGB = img[:,:,::-1]
    plt.subplot(121)
    plt.title(f"The total number of object: {len(contours)}")
    plt.imshow(imgRGB)

    imgSize = img.copy()
    imgShape = img.copy()
    for i, (size, contour) in enumerate(sorted_size_shape_list):
        # Compute the moment of contour:
        perimeter = cv.arcLength(contour, True)
        epsilon = 0.03 * perimeter
        approxPolyDP = cv.approxPolyDP(contour, epsilon, True)

        color = (0, 255, 255)
        thickness = 5
        # draw line
        for approx in approxPolyDP:
            cv.drawContours(imgSize, [approx], 0, color, thickness)
            cv.drawContours(imgShape, [approx], 0, color, thickness)
        color = (0, 0, 255)
        thickness = 5
        # draw points
        for approx in [approxPolyDP]:
            # draw points
            squeeze = np.squeeze(approx)
            #print('contour:',approx.shape, squeeze.shape)
            for p in squeeze:
                pp = tuple(p.reshape(1, -1)[0])
                cv.circle(imgSize, pp, 10, color, -1)
                cv.circle(imgShape, pp, 10, color, -1)
    
        # determine shape   
        verticeNumber = len(approxPolyDP)
        if verticeNumber == 3:
            vertice_shape = (verticeNumber, 'Triangle')
        elif verticeNumber == 4:
            # get aspect ratio
            x, y, width, height = cv.boundingRect(approxPolyDP)
            aspectRatio = float(width) / height
            #print(aspectRatio)
            if 0.90 < aspectRatio < 1.1: 
                vertice_shape = (verticeNumber, 'Square')
            else:
                vertice_shape = (verticeNumber, 'Rectangle')
        elif verticeNumber == 5:
            vertice_shape = (verticeNumber, 'Pentagon')
        elif verticeNumber == 6:
            vertice_shape = (verticeNumber, 'Hexagon')
        else:
            vertice_shape = (verticeNumber, 'Circle')
        
        M = cv.moments(contour)

        # The center or centroid can be calculated as follows:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Get the position to draw: 
        text1 = str(i + 1)   
        text2 = vertice_shape[1]
        textS = text1 + text2
        fontFace = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 3
        text_size = cv.getTextSize(textS, fontFace, fontScale, thickness)[0]

        text_x = cX - text_size[0] / 2
        text_x = round(text_x)
        text_y = cY + text_size[1] / 2
        text_y = round(text_y)
        
        # Write the ordering of the shape on the center of shapes
        color = (0, 0, 0)
        cv.putText(imgSize, textS, (text_x, text_y), fontFace, fontScale, color, thickness)
 
    # BGR to RGB

    imgRGB = imgSize[:,:,::-1]
    plt.subplot(122)
    plt.title("Sorted by Size then Shape")
    plt.imshow(imgRGB)

    plt.show()

def q1c():
    img = cv.imread(r'D:/KOTOCLASS/firstmonth/cv-master/essential/mypaintshape.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image:
    ret, threshImg = cv.threshold(imgGray, 150, 255, cv.THRESH_BINARY_INV)

    # Find contours using the thresholded image:
    contours, hierarchy = cv.findContours(threshImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    plt.figure(figsize=(10,10))

    imgRGB = img[:,:,::-1]
    plt.subplot(121)
    plt.title(f"The total number of object: {len(contours)}")
    plt.imshow(imgRGB)

    tri_count = 0
    rect_count = 0
    squ_count = 0
    pen_count = 0
    hex_count = 0
    cir_count = 0

    # approxPolyDP():
    imgApproxPolyDP = img.copy()
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        epsilon = 0.03 * perimeter
        approxPolyDP = cv.approxPolyDP(contour, epsilon, True)

        color = (0, 255, 255)
        thickness = 5
        
        # draw line
        for approx in approxPolyDP:
            cv.drawContours(imgApproxPolyDP, [approx], 0, color, thickness)
        color = (0, 0, 255)
        thickness = 5
        # draw points
        for approx in [approxPolyDP]:
            # draw points
            squeeze = np.squeeze(approx)
            #print('contour:',approx.shape, squeeze.shape)
            for p in squeeze:
                pp = tuple(p.reshape(1, -1)[0])
                cv.circle(imgApproxPolyDP, pp, 10, color, -1)
                
        # write shape
        # Compute the moment of contour:
        M = cv.moments(contour)

        # The center or centroid can be calculated as follows:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        # determine shape   
        verticeNumber = len(approxPolyDP)
        if verticeNumber == 3:
            vertice_shape = (verticeNumber, 'Triangle')

            tri_count+=1
            textTri = str(tri_count)
            textShape = vertice_shape[1]
            text1 = textTri + textShape
            fontFace = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 3
            text_size = cv.getTextSize(text1, fontFace, fontScale, thickness)[0]

            text_x = cX - text_size[0] / 2
            text_x = round(text_x)
            text_y = cY + text_size[1] / 2
            text_y = round(text_y)
            
            # Write the ordering of the shape on the center of shapes
            color = (0, 0, 0)
            cv.putText(imgApproxPolyDP, text1, (text_x, text_y), fontFace, fontScale, color, thickness)
            
        elif verticeNumber == 4:
            # get aspect ratio
            x, y, width, height = cv.boundingRect(approxPolyDP)
            aspectRatio = float(width) / height
            #print(aspectRatio)
            if 0.90 < aspectRatio < 1.1: 
                vertice_shape = (verticeNumber, 'Square')

                squ_count+=1
                textSqu = str(squ_count)
                textShape = vertice_shape[1]
                text2 = textSqu + textShape
                fontFace = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 3
                text_size = cv.getTextSize(text2, fontFace, fontScale, thickness)[0]

                text_x = cX - text_size[0] / 2
                text_x = round(text_x)
                text_y = cY + text_size[1] / 2
                text_y = round(text_y)
                
                # Write the ordering of the shape on the center of shapes
                color = (0, 0, 0)
                cv.putText(imgApproxPolyDP, text2, (text_x, text_y), fontFace, fontScale, color, thickness)
            else:
                vertice_shape = (verticeNumber, 'Rectangle')

                rect_count +=1
                textRect = str(rect_count)
                textShape = vertice_shape[1]
                text3 = textRect + textShape
                fontFace = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 3
                text_size = cv.getTextSize(text3, fontFace, fontScale, thickness)[0]

                text_x = cX - text_size[0] / 2
                text_x = round(text_x)
                text_y = cY + text_size[1] / 2
                text_y = round(text_y)
                
                # Write the ordering of the shape on the center of shapes
                color = (0, 0, 0)
                cv.putText(imgApproxPolyDP, text3, (text_x, text_y), fontFace, fontScale, color, thickness)

        elif verticeNumber == 5:
            vertice_shape = (verticeNumber, 'Pentagon')

            pen_count +=1
            textPen = str(pen_count)
            textShape = vertice_shape[1]
            text4 = textPen + textShape
            fontFace = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 3
            text_size = cv.getTextSize(text4, fontFace, fontScale, thickness)[0]

            text_x = cX - text_size[0] / 2
            text_x = round(text_x)
            text_y = cY + text_size[1] / 2
            text_y = round(text_y)
            
            # Write the ordering of the shape on the center of shapes
            color = (0, 0, 0)
            cv.putText(imgApproxPolyDP, text4, (text_x, text_y), fontFace, fontScale, color, thickness)

        elif verticeNumber == 6:
            vertice_shape = (verticeNumber, 'Hexagon')

            hex_count +=1
            textHex = str(hex_count)
            textShape = vertice_shape[1]
            text5 = textHex + textShape
            fontFace = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 3
            text_size = cv.getTextSize(text5, fontFace, fontScale, thickness)[0]

            text_x = cX - text_size[0] / 2
            text_x = round(text_x)
            text_y = cY + text_size[1] / 2
            text_y = round(text_y)
            
            # Write the ordering of the shape on the center of shapes
            color = (0, 0, 0)
            cv.putText(imgApproxPolyDP, text5, (text_x, text_y), fontFace, fontScale, color, thickness)

        else:
            vertice_shape = (verticeNumber, 'Circle')

            cir_count +=1
            textCir = str(cir_count)
            textShape = vertice_shape[1]
            text5 = textCir + textShape
            fontFace = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 3
            text_size = cv.getTextSize(text5, fontFace, fontScale, thickness)[0]

            text_x = cX - text_size[0] / 2
            text_x = round(text_x)
            text_y = cY + text_size[1] / 2
            text_y = round(text_y)
            
            # Write the ordering of the shape on the center of shapes
            color = (0, 0, 0)
            cv.putText(imgApproxPolyDP, text5, (text_x, text_y), fontFace, fontScale, color, thickness)


    imgRGB = imgApproxPolyDP[:,:,::-1]
    plt.subplot(122)
    plt.title("Sorted by Shape then Size")
    plt.imshow(imgRGB)

    plt.show()

def q1d():
    #filePath = sys.argv[1]
    # Load image:
    image = cv.imread(r"../CV-MASTER/essential/mypaintshape.png")
    #convert image into greyscale mode
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #find threshold of the image
    _, thrash = cv.threshold(gray_image, 240, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


    def find_sqr_rect(image): 
        for contour in contours:
            shape = cv.approxPolyDP(contour, 0.02*cv.arcLength(contour, True), True)
            x_cor = shape.ravel()[0]
            y_cor = shape.ravel()[1]
            
            if len(shape) ==4:
                #shape cordinates
                x,y,w,h = cv.boundingRect(shape)

                #width:height
                aspectRatio = float(w)/h
                cv.drawContours(image, [shape], 0, (0,255,0), 4)
                if 0.90 < aspectRatio < 1.1:
                    cv.putText(image, "Square", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
                else:
                    cv.putText(image, "Rectangle", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0))
                
        cv.imshow("Square and Rectangles", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_triangle(image):
        for contour in contours:
            shape = cv.approxPolyDP(contour, 0.26*cv.arcLength(contour, True), True)
            x_cor = shape.ravel()[0]
            y_cor = shape.ravel()[1]

            #For triangle
            if len(shape) ==3:
                cv.drawContours(image, [shape], 0, (0,255,0), 4)
                cv.putText(image, "Triangle", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))

                    
        cv.imshow("Triangles", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_pentagon(image):
        for contour in contours:
            shape = cv.approxPolyDP(contour, 0.02*cv.arcLength(contour, True), True)
            x_cor = shape.ravel()[0]
            y_cor = shape.ravel()[1]

            #For pentagon
            if len(shape) ==5:
                cv.drawContours(image, [shape], 0, (0,255,0), 4)
                cv.putText(image, "Pentagon", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))

                    
        cv.imshow("Pentagon", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_circle(image):
        for contour in contours:
            shape = cv.approxPolyDP(contour, 0.01*cv.arcLength(contour, True), True)
            x_cor = shape.ravel()[0]
            y_cor = shape.ravel()[1]-15
            
            if len(shape) >12:
                cv.drawContours(image, [shape], 0, (0,255,0), 4)
                cv.putText(image, "Circle", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))

                    
        cv.imshow("circle", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_hexagon(image):
        for contour in contours:
            shape = cv.approxPolyDP(contour, 0.02*cv.arcLength(contour, True), True)
            x_cor = shape.ravel()[0]
            y_cor = shape.ravel()[1]
            
            if len(shape) == 6:
                cv.drawContours(image, [shape], 0, (0,255,0), 4)
                cv.putText(image, "Hexagon", (x_cor, y_cor), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))

                    
        cv.imshow("Hexagon", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if __name__ == "__main__":
        img = image.copy()
        user_input = input("""Shape detection optionb:
                            a : Detect Squares and Rectangles
                            b : Detect Triangles
                            c : Detect Pentagons
                            d : Detect Hexagon
                            e : Detect Circles
                            """)

        if user_input == 'a' :
            find_sqr_rect(img)
        elif user_input == 'b':
            find_triangle(img)
        elif user_input == 'c':
            find_pentagon(img)
        elif user_input == 'd':
            find_hexagon(img)
        elif user_input == 'e':
            find_circle(img)

def q2_image(filePath):
    #filePath = sys.argv[1]
    # Load image:
    #img = cv.imread(filePath)
    img = cv.imread(r"D:/KOTOCLASS/firstmonth/cv-master/essential/assets/faceAssignment.jpg")

    # image dimension
    h, w = img.shape[:2]

    # load model
    model = cv.dnn.readNetFromTensorflow(r"D:/KOTOCLASS/firstmonth/cv-master/samples/data/opencv_face_detector_uint8.pb", r"D:/KOTOCLASS/firstmonth/cv-master/samples/data/opencv_face_detector.pbtxt")

    # preprocessing
    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [
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
        if confidence > 0.9:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            text = "{:.3f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(img, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show
    fig = plt.figure(figsize=(20, 10))

    # Plot the images:
    imgRGB = img[:,:,::-1]
    plt.title(f'The total number of face detect: {faceCounter}')
    plt.imshow(imgRGB)

    plt.show()

def q2_video():
    capture = cv.VideoCapture(0)

    # check if connected
    #filePath = filePath = sys.argv[1]
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
        h, w = facedetect.shape[:2]

        # load model
        model = cv.dnn.readNetFromTensorflow(r"D:/KOTOCLASS/firstmonth/cv-master/samples/data/opencv_face_detector_uint8.pb",r"D:/KOTOCLASS/firstmonth/cv-master/samples/data/opencv_face_detector.pbtxt")

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

if __name__ == "__main__":
    user_input = input("""Please select:
                        a : Sort by Size and Shape
                        b : Sort by Size then Shape
                        c : Sort by Shape and Size
                        d : Detect certain shape
                        e : Detect faces in an image
                        f : Detect faces in video
                        """)

    if user_input == 'a' :
        q1a()
    elif user_input == 'b':
        q1b()
    elif user_input == 'c':
        q1c()
    elif user_input == 'd':
        q1d()
    elif user_input == 'e':
        filePath = input("Image directory: ")
        q2_image(filePath)
    elif user_input == 'f':
        q2_video()
