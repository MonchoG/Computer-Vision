import pickle

import cv2
import face_recognition
from cv2.data import haarcascades as haarcascade

print("[INFO] loading encodings + face detector...")

data = pickle.loads(open("encodings.pickle", "rb").read())


def encode_faces(image, boxes, reSample = 1):
    print(" [ INFO ] Starting encoding")
    encodings = face_recognition.face_encodings(image, boxes, reSample)
    print(" [ INFO ] Done encoding")
    return encodings


def find_faces(image, encodings, boxes, color):
    print(" [ INFO ] Starting find_faces")
    names = []
    (b, g, r) = color
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding, 0.55)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            print(" [ INFO ] Founds matches")
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
                    0.75, (b, g, r), 2)

    print(" [ INFO ] Returning matches")

    return image


def extract_face(image, name):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box_frontal = get_boxes('frontal_face', gray)
    box_profile = get_boxes('profile_face', gray)

    if box_frontal:
        cascade = 'frontal'
        box = box_frontal

    elif box_profile:
        cascade = 'profile'
        box = box_profile

    else:
        cascade = 'profile_flipped'
        box = get_boxes('profile_face_flipped', gray)

    for face in box:
        (top, right, bottom, left) = face
        face_roi = img[top:bottom, left:right]
        cv2.imwrite('face_{}_{}.jpg'.format(cascade, name), face_roi)


def get_boxes(detect_area, image, scaleFactor=1.1, minNeighbors=40, minSize=(35, 35)):
    cascade = ''
    if detect_area.__contains__('profile_face'):
        cascade = "haarcascade_profileface.xml".format(haarcascade)

    elif detect_area.__contains__('frontal_face'):
        cascade = "haarcascade_frontalface_default.xml".format(haarcascade)

    elif detect_area.__contains__('profile_flip'):
        image = cv2.flip(image, +1)
        cascade = "haarcascade_profileface.xml".format(haarcascade)

    print(" [ INFO ] Starting detector : {}".format(cascade))
    detector = cv2.CascadeClassifier('{}/{}'.format(haarcascade, cascade))
    rects = detector.detectMultiScale(image, scaleFactor=scaleFactor,
                                      minNeighbors=minNeighbors, minSize=minSize)
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    print(" [ INFO ] Finished detector : num of boxes {}".format(len(boxes)))

    return boxes


# The same as get boxes, but this uses face_recognition NN to detect
def compute_face_locations(image, model):
    print("[ INFO ] Computing face locations")
    locations = face_recognition.face_locations(image, model='{}'.format(model))
    print("[ INFO ] Finished Computing face locations")

    return locations
