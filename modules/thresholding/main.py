from modules.morphology.morphology import *


## Timer
# call the functions to start/stop timer
# elapsed returns the time spent
##
def start():
    return timer()


def stop():
    return timer()


def elapsed(t_begin, t_end):
    return t_end - t_begin


## find_countours
# @param:
# image - image object
# title - string to save the image
##
def find_countours(image, title):
    thrsh = opencv_thresh_otsu(image, 60)
    countours = cv2.findContours(thrsh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    countours = imutils.grab_contours(countours)

    largest_area = [0, ]
    previous_area = 0

    lowest_eccentricity = []
    previous_eccentricity = 0
    previous_lowest = 0
    counter = 0

    for countour in countours:
        # if the contour is not sufficiently large, ignore it
        counter += 1
        print("\n==== Counter : {}====".format(counter))
        area = cv2.contourArea(countour)

        if area < 150:
            continue

        print("== Calculating center of mass")
        M = cv2.moments(countour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print("== cX : {}  |  cY : {}".format(cX, cY))
        # draw the contour and center of the shape on the image
        cv2.line(image, (cX, cY - 10), (cX, cY + 10), (0, 0, 255), 2, -1)
        cv2.line(image, (cX - 10, cY), (cX + 10, cY), (0, 0, 255), 2, -1)

        print("== Last Area : {}".format(previous_area))
        print("== Current Area :  {}".format(area))

        if area >= largest_area[0]:
            (x, y, w, h) = cv2.boundingRect(countour)
            largest_area = [area, (x, y), (x + w, y + h), [cX, cY]]
            print("== Found larger Area")

        previous_area = area

        (x_centre, y_centre), (minor_axis, major_axis), angle = cv2.fitEllipse(countour)
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))

        print("== Last Eccentricity : {}".format(previous_eccentricity))
        print("== Current Eccentricity : {}".format(eccentricity))
        # print("== Current  Lowest Eccentricity : {}".format(lowest_eccentricity[0]))

        if previous_lowest == 0:
            previous_lowest = eccentricity

        elif eccentricity < previous_lowest:
            (x, y, w, h) = cv2.boundingRect(countour)
            lowest_eccentricity = [eccentricity, (x, y), (x + w, y + h), [x_centre, y_centre]]
            previous_lowest = eccentricity
            print("== Found lower eccentricity")
        previous_eccentricity = eccentricity

        cv2.putText(image, "center : {}".format(counter), (cX + 20, cY - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, "Area : {:06.2f}".format(area),
                    (cX + 20, cY + 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, "Eccentricity : {:06.4f} ".format(eccentricity),
                    (cX + 20, cY + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

    print('== Largest Area : {}'.format(largest_area[0]))
    cv2.rectangle(image, (largest_area[1]), (largest_area[2]), (0, 255, 0), 2)

    print('== Lowest Eccentricity : {}'.format(lowest_eccentricity[0]))
    cv2.rectangle(image, (lowest_eccentricity[1]), (lowest_eccentricity[2]), (255, 0, 0), 2)

    cv2.imwrite('{}.jpg'.format(title), image)

    return image


## opencv_thresh_otsu
# @param:
# image - image object
# thrsh - initial threshold
# @return:
#   thresholded image
##
def opencv_thresh_otsu(img, thrsh):
    print("=============== Starting job - Opencv thresh {} ===============".format(thrsh))
    t_start = start()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    ret1, thresh1 = cv2.threshold(blurred, thrsh, 255, cv2.THRESH_OTSU)

    print("Elapsed time: {}".format(elapsed(t_start, stop())))
    cv2.imwrite("opencv_thresh_otsu_{}.png".format(thrsh), thresh1)
    return thresh1


## thresh_optimized
# @param:
# image - image object
# thresh - initial threshold
# const - error constant
# @return:
#   thresholded image
##
def thresh_optimized(img, thresh, const):
    print("=============== Starting job - thresh optimized {} ===============".format(thresh))
    t_start = start()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    initial_thresh = thresh
    iteration = 0
    height, widght = gray_img.shape[:2]
    pixels = []
    for y in range(height):
        for x in range(widght):
            pixels.append(gray_img[y, x])

    while True:
        iteration += 1
        print("Starting iteration: {}".format(iteration))

        seg_below = []
        seg_above = []
        for pixel in pixels:
            if pixel > thresh:
                seg_above.append(pixel)
            elif pixel <= thresh:
                seg_below.append(pixel)
        avg_below = np.mean(seg_below)
        avg_above = np.mean(seg_above)

        print("Avg. Below: {} ------  Avg. Above: {}".format(avg_below, avg_above))

        new_thresh = (avg_below + avg_above) * 0.5
        delta_t = abs(thresh - new_thresh)
        print("Old Threshold Value: {}".format(thresh))
        print("New Threshold Value: {}".format(new_thresh))
        print("Delta T: {}".format(delta_t))

        thresh = new_thresh
        if delta_t <= const:
            break
    print("final_thresh: {}".format(thresh))

    for pixel in pixels:
        if pixel > thresh:
            pixel = 255
        elif pixel <= thresh:
            pixel = 0

    cv2.imwrite("thresholded_optimized_{}_{}.png".format(initial_thresh, thresh), gray_img)
    time = elapsed(t_start, stop())
    print("Elapsed time: {}".format(time))

    return thresh, time


## thresh
# @param:
# image - image object
# thresh - initial threshold
# const - error constant
# @return:
#   thresholded image
##
def thresh(img, thresh, const):
    print("=============== Starting job - thresh {} ===============".format(thresh))
    t_start = start()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    initial_thresh = thresh
    iteration = 0
    while True:
        iteration += 1
        print("Starting iteration: {}".format(iteration))
        height, widght = gray_img.shape[:2]
        seg_below = []
        seg_above = []
        for y in range(height):
            for x in range(widght):
                pixel = gray_img[y, x]
                if pixel > thresh:
                    seg_above.append(gray_img[y, x])
                elif pixel <= thresh:
                    seg_below.append(gray_img[y, x])
        avg_below = np.mean(seg_below)
        avg_above = np.mean(seg_above)

        print("Avg. Below: {} ------  Avg. Above: {}".format(avg_below, avg_above))

        new_thresh = (avg_below + avg_above) * 0.5
        delta_t = abs(thresh - new_thresh)
        print("Old Threshold Value: {}".format(thresh))
        print("New Threshold Value: {}".format(new_thresh))
        print("Delta T: {}".format(delta_t))

        thresh = new_thresh
        if delta_t <= const:
            break

    print("final_thresh: {}".format(thresh))

    for y in np.arange(height):
        for x in np.arange(widght):
            pixel = gray_img[y, x]

            if pixel > thresh:
                gray_img[y, x] = 255
            elif pixel <= thresh:
                gray_img[y, x] = 0

    cv2.imwrite("thresholded_{}_{}.png".format(initial_thresh, thresh), gray_img)
    time = elapsed(t_start, stop())
    print("Elapsed time: {}".format(time))

    return thresh, time


def __main__():
    frame = cv2.imread("objects.jpg")

    thresh1, time1 = thresh(frame, 40, 0.5)
    thresh4, time2 = thresh(frame, 127, 0.5)
    thresh8, time3 = thresh(frame, 200, 0.5)

    thresh1_optimized, time_optimized_1 = thresh_optimized(frame, 40, 0.5)
    thresh4_optimized, time_optimized_2 = thresh_optimized(frame, 127, 0.5)
    thresh8_optimized, time_optimized_3 = thresh_optimized(frame, 200, 0.5)

    opencv_thresh_otsu(frame, 40)
    opencv_thresh_otsu(frame, 127)
    opencv_thresh_otsu(frame, 200)

    print("   ==  normal ==  optimized  ==\n"
          "{} ==  {}  ==  {}  ==\n"
          "{} ==  {}  ==  {}  ==\n"
          "{} ==  {}  ==  {}  ==\n".format(40, time1, time_optimized_1, 127, time2, time_optimized_2, 200, time3,
                                           time_optimized_3))

    find_countours(frame, "countours_test")


__main__()
