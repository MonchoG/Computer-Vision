import base64
import pickle
import socket
import struct
import sys
import time
from _thread import start_new_thread

import cv2

import modules.face_recognition.recognize_faces as recognizer


def string_to_image(string, title):
    fh = open("{}.png".format(title), "wb")
    fh.write(base64.b64decode(string))
    print("Done writing file.")
    fh.close()


HOST = '192.168.178.208'
PORT = 9999

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error as msg:
    print("Could not create socket. Error Code: ", str(msg[0]), "Error: ", msg[1])
    sys.exit(0)

print("[-] Socket Created")

# bind socket
try:
    s.bind((HOST, PORT))
    print("[-] Socket Bound to host " + str(HOST))
    print("[-] Socket Bound to port " + str(PORT))
except socket.error as msg:
    print("Bind Failed. Error Code: {} Error: {}".format(str(msg[0]), msg[1]))
    sys.exit()

s.listen(10)
print("Listening...")


# Client handler
def client_thread(conn, addr):
    try:
        recv_0 = 0
        data = b""
        payload_size = struct.calcsize(">L")
        print("payload_size: {}".format(payload_size))
        while True:
            while len(data) < payload_size:
                print("Recv: {} from {}".format(len(data), addr[0]))
                data += conn.recv(4096)

            if len(data) == 0:
                recv_0 += 1

            # if recv_0 >= 1000:
            #     break

            print("Done Recv: {} from {}".format(len(data), addr[0]))
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            print("msg_size: {}".format(msg_size))
            while len(data) < msg_size:
                data += conn.recv(4096)

            if str(data).__contains__('Bye'):
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)

            cv2.imwrite('{}.png'.format(timestamp), frame)
            result = recognizer.find_face(frame)
            cv2.imwrite('result_{}.png'.format(timestamp), result)
            time.sleep(20)

    except Exception as exp:
        print(exp)
        print(exp.with_traceback())
        exit()

    finally:
        print("Exited all loops closing connections")
        conn.close()


while True:
    # blocking call, waits to accept a connection
    try:
        conn, addr = s.accept()
        print("[-] Connected to " + addr[0] + ":" + str(addr[1]))

        start_new_thread(client_thread, (conn, addr))
    except socket.error as e:
        print("Error occurred during initializing connection")
        print(e.strerror)
