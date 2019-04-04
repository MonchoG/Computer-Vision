# import the necessary packages
import cv2
import imutils

import modules.face_recognition.image_tool as tool


def __main__():
    cam = cv2.VideoCapture(0)
    # initialize the video stream and allow the camera sensor to warm
    print("[INFO] starting opening image...")
    while True:
        # # Capture frame-by-frame
        ret, image = cam.read()
        if ret:
            image = imutils.resize(image, width=640)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # tool.extract_face(image, '{}'.format(iteration))

            # detect faces in the grayscale frame
            rects = tool.get_boxes('profile_face', gray)
            rects2 = tool.get_boxes('frontal_face', gray)
            rects3 = tool.get_boxes('profile_flip', gray)

            # compute the facial embeddings for each face bounding box
            if rects:
                encodings = tool.encode_faces(rgb, rects)
                image = tool.find_faces(image, encodings, rects, (255, 0, 0))
            if rects2:
                encodings2 = tool.encode_faces(rgb, rects2)
                image = tool.find_faces(image, encodings2, rects2, (0, 255, 0))
            if rects3:
                encodings3 = tool.encode_faces(rgb, rects3)
                image = tool.find_faces(image, encodings3, rects3, (0, 0, 255))

            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                exit(0)


__main__()
