import os
import pickle

import cv2
from imutils import paths

import modules.face_recognition.image_tool as tool

imagePaths = list(paths.list_images("dataset"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("[INFO] computing face locations ")
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = tool.compute_face_locations(rgb, 'hog')
    print("[INFO] Encoding face")
    # compute the facial embedding for the face
    encodings = tool.encode_faces(rgb, boxes, reSample=20)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)
    print("[INFO] Done processing image {}/{}. ".format(i + 1,
                                                        len(imagePaths)))
    # dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "rb+")
f.write(pickle.dumps(data))
f.close()
