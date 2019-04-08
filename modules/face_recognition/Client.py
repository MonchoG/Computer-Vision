import pickle
import socket
import struct
import time

import cv2
import picamera.array

# initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()


# allow the camera to warmup
time.sleep(1.0)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.178.208', 9999))
connection = client_socket.makefile('wb')

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
with picamera.PiCamera() as camera:
    camera.resolution = (2592, 1952)
    camera.framerate = 15
    #camera.start_preview()
    time.sleep(2)
    while True:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            # At this point the image is available as stream.array
            image = stream.array
            result, frame = cv2.imencode('.jpg', image, encode_param)
            # show the frame
            # ata = zlib.compress(pickle.dumps(frame, 0))
            data = pickle.dumps(frame, 0)
            size = len(data)
            print("{}: {}".format(img_counter, size))
            client_socket.sendall(struct.pack(">L", size) + data)
            img_counter += 1
            time.sleep(20)
#            if img_counter == 10:
#                data = 'Bye'
#                size = len(data)
#                print("{}: {}".format(img_counter, size))
#                client_socket.sendall(struct.pack(">L", size) + bytes(data))

