import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

car_cascade = cv2.CascadeClassifier('cars.xml')

while cap.isOpened():
    success, img = cap.read()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    if success:
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('video2', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break