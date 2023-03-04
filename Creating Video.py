def Take_Pic():
    import cv2

    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    result, image = cam.read()


    if result:
        cv2.imshow("Hacktech2023", image)


        cv2.imwrite("Hacktech2023", image)

        cv2.destroyWindow("Hacktech2023")



