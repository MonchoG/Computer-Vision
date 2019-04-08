# import the necessary packages
import cv2
import imutils
import modules.face_recognition.image_tool as tool


def find_face(image):
    image = imutils.resize(image, width=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    (b, g, r) = (0, 0, 0)
    rects = tool.get_boxes('frontal_face', gray)
    # compute the facial embeddings for each face bounding box
    if rects:
        (b, g, r) = (255, 0, 0)
    else:
        rects = tool.get_boxes('profile_face', gray)
        if rects:
            (b, g, r) = (0, 255, 0)
        else:
            rects = tool.get_boxes('profile_flip', gray)
            if rects:
                (b, g, r) = (0, 0, 255)
            else:
                print("No face Detected")

    encoding = tool.encode_faces(rgb, rects)
    image = tool.find_faces(image, encoding, rects, (b, g, r))

    return image
