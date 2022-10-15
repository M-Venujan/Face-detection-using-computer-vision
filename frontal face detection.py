#import computer vision library
import cv2 as cv
# Load the trained algorithm
trained_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the image
img = cv.imread('people.jpg')
# Convert into grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Detect faces using algorithm
face_coordinates = trained_data.detectMultiScale(gray)
# Getting requires point for array
for (x, y, w, h) in face_coordinates:
    #getting points for top left corner and adding width and height to make rectangle
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv.imshow('img', img)
cv.waitKey(10000)
