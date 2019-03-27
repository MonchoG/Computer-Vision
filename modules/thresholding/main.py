import cv2
import numpy as np
from timeit import default_timer as timer


# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
#
# plt.show()

def start():
    t = timer()

    return t


def stop():
    t = timer()
    return t


def elapsed(t_begin, t_end):
    return t_end - t_begin\

def thresholding(image, threshold):
    print("Starting job - thresh {}".format(threshold))
    t_start = start()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    initial_thresh = threshold
    last_thresh = 0

    (m1, m2) = image.shape[:2]
    print(m1)
    print(m2)
    while True:
        pixels = []
        segment_below = []
        segment_above = []
        for y in np.arange(m1):
            for x in np.arange(m2):
                pixel = image[y, x]
                pixels.append([y, x])
                if pixel < threshold:
                    segment_below.append(image[y, x])
                elif pixel > threshold:
                    segment_above.append(image[y, x])

        mean_below = np.mean(segment_below)
        mean_above = np.mean(segment_above)

        threshold = (mean_below + mean_above) / 2
        if threshold < initial_thresh:
            print("exiting thresh - thresh is {}".format(threshold))
            break
        elif last_thresh == threshold:
            break
        else:
            last_thresh = threshold
            print(threshold)
            print("repeating")

    for coordinate in pixels:
        pixel = image[coordinate[0], coordinate[1]]
        if pixel > threshold:
            image[coordinate[0], coordinate[1]] = 255

        elif pixel <= threshold:
            image[coordinate[0], coordinate[1]] = 0

    cv2.imwrite("thresholded_{}.png".format(initial_thresh), image)

    print("Elapsed time: {}".format(elapsed(t_start, stop())))


def __main__():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    cv2.imwrite("image.png", frame)

    thresholding(frame, 20)
    thresholding(frame, 40)
    thresholding(frame, 60)
    thresholding(frame, 80)
    thresholding(frame, 127)
    thresholding(frame, 140)
    thresholding(frame, 160)
    thresholding(frame, 180)
    thresholding(frame, 200)
    thresholding(frame, 227)
    cam.release()


__main__()
