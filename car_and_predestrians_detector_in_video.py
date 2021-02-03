import cv2

# our video
video = cv2.VideoCapture('Self_Driving_1.mp4')

# our pre-trained car and predestrians files
car_cls_file = 'car_detector.xml'
predestrians_cls_file = 'haarcascade_fullbody.xml'

# create car and predestrians classifier
car_tracker = cv2.CascadeClassifier(car_cls_file)
predestrians_tracker = cv2.CascadeClassifier(predestrians_cls_file)

# Run forever..
while True:
    # Read the current frame from video
    (read_successful, frame) = video.read()

    # safe coding
    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and predestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    predestrians = predestrians_tracker.detectMultiScale(grayscale_frame)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    # Draw rectangles around the predestrians
    for (x, y, w, h) in predestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image with the faces spotted
    cv2.imshow('Car & Predestrians Detector', frame)

    # Dont autoclose
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


# releasing the videoCapture frame..
video.release()
