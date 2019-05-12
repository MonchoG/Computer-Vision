import cv2
import face_recognition


# face_cascade = cv2.CascadeClassifier('{}/haarcascade_frontalface_default.xml'.format(haarcascades))
# profile_face_cascade = cv2.CascadeClassifier('{}/haarcascade_frontalface_default.xml'.format(haarcascades))
# eye_cascade = cv2.CascadeClassifier('{}/haarcascade_eye.xml'.format(haarcascades))


def load_image(path):
    return face_recognition.load_image_file(path)


# Returns a list with location of faces
def detect_faces(img):
    try:
        list = face_recognition.face_locations(img)
        if len(list) > 0:
            return True, list
        else:
            return False, []
    except Exception as e:
        print(e)
        # exit(0)


# Encode Face
def encode_face(face):
    return face_recognition.face_encodings(face)[0]


# Detect multiple faces and encode them return a list with face encodings
def multiple_faces(image):
    face_locations = face_recognition.face_locations(image)
    print("Found {} faces in image.".format(len(face_locations)))
    encodings_list = []
    encodings = face_recognition.face_encodings(image, face_locations)
    for face_encoding in encodings:
        encodings_list.append(face_encoding)
    return encodings_list


# get landmarks / return a list
def face_landmarks(face_image):
    try:
        landmarks = face_recognition.face_landmarks(face_image)
        return landmarks

    except Exception as e:
        print(e)
        exit(0)


def compare_faces(list_encodings, face_encoding):
    return face_recognition.compare_faces(list_encodings, face_encoding)


def draw_rectangles(img):
    flag, faces = detect_faces(img)
    face_num = 0
    if flag:
        for (top, right, bottom, left) in faces:
            face_num += 1
            face = cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        return img, faces

# moncho1 = cv2.imread('moncho1.png')
# moncho2 = cv2.imread('moncho2.png')
# moncho3 = cv2.imread('moncho3.png')
#
# milcho1 = cv2.imread('milcho1.png')
#
# flag, list = detect_faces(moncho1)
# moncho2_detect = detect_faces(moncho2)
# moncho3_detect = detect_faces(moncho3)
#
# milcho1_detect = detect_faces(milcho1)
#
# print("moncho1 locations {}".format(moncho1_detect))
# print("moncho2 locations {}".format(moncho2_detect))
# print("moncho3 locations {}".format(moncho3_detect))
#
# print("milcho1 locations {}".format(milcho1_detect))
#
# moncho1_features = face_landmarks(moncho1)
# moncho2_features = face_landmarks(moncho2)
# moncho3_features = face_landmarks(moncho3)
#
# milcho1_features = face_landmarks(milcho1)
#
# print("moncho1 features {}".format(moncho1_features))
# print("moncho2 features {}".format(moncho2_features))
# print("moncho3 features {}".format(moncho3_features))
#
# print("milcho1 features {}".format(milcho1_features))
#
# moncho1_encode = encode_face(moncho1)
# moncho2_encode = encode_face(moncho2)
# moncho3_encode = encode_face(moncho3)
#
# milcho1_encode = encode_face(milcho1)
#
# print("moncho1 encode {}".format(moncho1_encode))
# print("moncho2 encode {}".format(moncho2_encode))
# print("moncho3 encode {}".format(moncho3_encode))
#
# print("milcho1 encode {}".format(milcho1_encode))

# moncho1_vs_moncho2 = face_recognition.compare_faces(moncho1_encode, moncho2_encode[0])
# moncho2_vs_moncho3 = face_recognition.compare_faces(moncho2_encode, moncho3_encode[0])
# moncho1_vs_moncho3 = face_recognition.compare_faces(moncho1_encode, moncho3_encode[0])
#
# moncho1_vs_milcho1 = face_recognition.compare_faces(moncho1_encode, milcho1_encode[0])
#
# print("moncho1_vs_moncho2 : {} |  moncho2_vs_moncho3 : {} | moncho1_vs_moncho3 : {} | moncho1 vs milcho1 : {}".format(
#     moncho1_vs_moncho2, moncho2_vs_moncho3, moncho1_vs_moncho3, moncho1_vs_milcho1))
#
# if cv2.waitKey(15) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cam.read()
#     if ret:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#         profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#
#         face_num = 0
#         profile_face_num = 0
#         print("Found {} faces! Found {} profiles".format(len(faces), len(profile_faces)))
#         # Draw a rectangle around the faces
#         for (x, y, w, h) in faces:
#             face_num += 1
#             img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             roi_color = img[y:y + h, x:x + w]
#             cv2.imwrite('face_{}.png'.format(face_num), img)
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#         for (x, y, w, h) in profile_faces:
#             profile_face_num += 1
#             img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             roi_color = img[y:y + h, x:x + w]
#             cv2.imwrite('face_{}.png'.format(face_num), img)
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#         cv2.imshow('Video', frame)
#
#         if cv2.waitKey(15) & 0xFF == ord('q'):
#             break
#
#     else:
#         break
# cam.release()
# # video.release()
# cv2.destroyAllWindows()
