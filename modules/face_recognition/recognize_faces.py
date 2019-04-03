# import the necessary packages
import cv2
import imutils

import modules.face_recognition.image_tool as tool


def __main__():
    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection

    cam = cv2.VideoCapture(1)
    iteration = 0
    # initialize the video stream and allow the camera sensor to warm
    print("[INFO] starting opening image...")
    while True:
        #     # Capture frame-by-frame
        ret, image = cam.read()
        iteration += 1
        if ret:

            image = imutils.resize(image, width=1280, height=720)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #tool.extract_face(image, '{}'.format(iteration))

            # detect faces in the grayscale frame
            rects = tool.get_boxes('profile_face', gray)
            rects2 = tool.get_boxes('frontal_face', gray)

            # compute the facial embeddings for each face bounding box
            encodings = tool.encode_faces(rgb, rects)
            encodings2 = tool.encode_faces(rgb, rects2)

            image = tool.find_faces(image, encodings, rects, (255, 0, 0))
            image = tool.find_faces(image, encodings2, rects2, (0, 255, 0))

            cv2.imshow("Frame", image)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                exit(0)


__main__()
