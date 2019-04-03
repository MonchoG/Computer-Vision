import pickle

import cv2
import face_recognition
from cv2.data import haarcascades as haarcascade

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open("encodings.pickle", "rb").read())


def encode_faces(image, boxes):
    return face_recognition.face_encodings(image, boxes)


def find_faces(image, encodings, boxes, color):
    names = []
    (b, g, r) = color
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom),
                      (b, g, r), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
    return image


def extract_face(image, name):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box_frontal = get_boxes('frontal_face', gray)
    box_profile = get_boxes('profile_face', gray)

    if box_profile:
        cascade = 'profile'
        box = box_profile

    else:
        cascade = 'frontal'
        box = box_frontal

    for face in box:
        (top, right, bottom, left) = face
        face_roi = img[top:bottom, left:right]
        cv2.imwrite('face_{}_{}.jpg'.format(cascade, name), face_roi)


def get_boxes(detect_area, image):
    cascade = ''
    if detect_area.__contains__('profile_face'):
        cascade = "haarcascade_profileface.xml".format(haarcascade)

    elif detect_area.__contains__('frontal_face'):
        cascade = "haarcascade_frontalface_default.xml".format(haarcascade)

    detector = cv2.CascadeClassifier('{}/{}'.format(haarcascade, cascade))
    rects = detector.detectMultiScale(image, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30))
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    return boxes


# The same as get boxes, but this uses face_recognition NN to detect
def compute_face_locations(image, model):
    return face_recognition.face_locations(image, model)
