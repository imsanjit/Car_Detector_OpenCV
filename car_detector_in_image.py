import cv2

# our image
img_file = 'car_image_1.jpg'

# our pre-trained car classifier
cls_file = 'car_detector.xml'

# create opencv image
img = cv2.imread(img_file)

# convert image to grayscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(cls_file)

# detect car
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(black_n_white, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with the faces spotted
cv2.imshow('Car Detector', black_n_white)

# Dont autoclose
cv2.waitKey()




print('Code Completed')
