import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert it to grayscale:
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
