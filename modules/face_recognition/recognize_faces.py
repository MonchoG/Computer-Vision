# import the necessary packages
import cv2
import imutils

import modules.face_recognition.image_tool as tool


def find_face(image, scaleFactor=None, minNeighbors=None, minSize=None, reSample=1):
   # image = imutils.resize(image, width=2592)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (b, g, r) = (0, 0, 0)
    rects = tool.get_boxes('frontal_face', gray, scaleFactor, minNeighbors, minSize)
    # compute the facial embeddings for each face bounding box
    if rects:
        (b, g, r) = (255, 0, 0)
    else:
        rects = tool.get_boxes('profile_face', gray, scaleFactor, minNeighbors, minSize)
        if rects:
            (b, g, r) = (0, 255, 0)
        else:
            rects = tool.get_boxes('profile_flip', gray, scaleFactor, minNeighbors, minSize)
            if rects:
                (b, g, r) = (0, 0, 255)
            else:
                print("No face Detected")

    encoding = tool.encode_faces(rgb, rects, reSample)
    image = tool.find_faces(image, encoding, rects, (b, g, r))

    return image


# img = cv2.imread('class.jpg')
# result = find_face(img, scaleFactor=1.1, minNeighbors=7, minSize=(35, 35), reSample=3)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
