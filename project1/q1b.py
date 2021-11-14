#Question 1 part 2

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
plt.figure(figsize=(15,10))

# BGR to RGB
imgRGB = img[:,:,::-1]
plt.subplot(221)
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
    text1 = vertice_shape[1]
    text2 = str(i + 1)
    text = text1 + text2
    text3 = text2 + text1
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
    cv.putText(imgShape, text3, (text_x, text_y), fontFace, fontScale, color, thickness)
    

# BGR to RGB
imgRGB = imgSize[:,:,::-1]
plt.subplot(223)
plt.title("Sorted by Shape then Size")

plt.imshow(imgRGB)

imgRGB = imgShape[:,:,::-1]
plt.subplot(224)
plt.title("Sorted by Size then Shape")
plt.imshow(imgRGB)

plt.show()