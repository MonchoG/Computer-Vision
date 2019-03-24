import cv2
import numpy as np
from matplotlib import pyplot as plt


###
#   @method:
#       -resize
#   @param:
#        - scale_percent
#        - image
#   @ return:
#       - resized image based on the scale percent
###
def resize(scale_percent, image):
    width = int(image.shape[1] * (scale_percent - 10) / 100)
    height = int(image.shape[0] * (scale_percent - 5) / 100)
    dim = (width, height)
    result = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return result


###
#   @method:
#       - draw_points
#   @param:
#       - img
#       - points
#   @return:
#       -returns image with points drawn on it
###
def draw_points(img, points):
    for point in points:
        cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

        cv2.putText(img, '[{},{}]'.format(point[0] / 10, point[1] / 10), (point[0], point[1]),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('dots', img)
    cv2.waitKey(0)
    return img


###
#   @method:
#       -  plot_img
#   @param:
#       -  img
#   @return
#       -  Plots and image to a pyplot. Useful to extract points coordinates
###
def plot_img(img):
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.show()


###
#   @method:
#       -put_on_background
#   @param:
#        - background
#        - foreground
#        - coordinate x
#        - coordinate y
#        - title
#   @ return:
#       - new image, put the foreground to the background image
###
def put_on_background(background, foreground, coordinate_x, coordinate_y, title):
    rows, cols, ch = foreground.shape

    trans_indices = foreground[..., 2] != 0
    overlay_copy = background[coordinate_y:coordinate_y + rows, coordinate_x:coordinate_x + cols]

    overlay_copy[trans_indices] = foreground[trans_indices]
    background[coordinate_y:coordinate_y + rows, coordinate_x:coordinate_x + cols] = overlay_copy

    cv2.imwrite("{}.png".format(title), background)

    return background


###
#   @method -
#       - affine_transform
#   @param -
#       - img - source image
#       - _src - source points
#       - _dst - destination points
#       - title - title of the image
#   @return
#       - returns the affine transformation of the image
###
def affine_transform(img, _src, _dst, title):
    rows, cols, ch = img.shape
    M = cv2.getAffineTransform(_src, _dst)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite("{}.png".format(title), dst)
    return dst


###
#   @method -
#       - perspective_transformation
#   @param -
#       - img - source image
#       - _src - source points
#       - _dst - destination points
#       - title - title of the plot
###
def perspective_transform(img, _src, _dst, title):
    M, mask = cv2.findHomography(_src, _dst)
    dst = cv2.warpPerspective(img, M, (560, 980))

    cv2.imshow("{}.png".format(title), dst)
    cv2.imwrite("{}.png".format(title), dst)

    cv2.waitKey(0)
    return dst


def __main__():
    # for cat in a phone
    background = cv2.imread("phone.png")
    foreground = cv2.imread("cat.png")
    foreground = resize(40, foreground)
    affined = affine_transform(foreground, np.float32([[60, 0], [5, 70], [60, 70]]),
                               np.float32([[40, 10], [15, 70], [50, 63]]), "cat_affined")
    img = put_on_background(background, affined, 75, 55, 'cat_in_the_phone')

    # # for tennis court
    tennis_court = cv2.imread("tennis_court.png")
    # TOP_LEFT_COURT, TOP_RIGHT_COURT, BOTTOM_LEFT_COURT, BOTTOM_RIGHT_COURT, MARGIN_TOP_LEFT, MARGIN_TOP_RIGHT, MARGIN_BOTTOM_LEFT, MARGIN_BOTTOM_RIGHT
    changed = perspective_transform(tennis_court, np.float32(
        [[200, 72], [585, 90], [22, 280], [865, 332], [180, 52], [605, 70], [2, 380], [865, 380]]), np.float32(
        [[100, 100], [460, 100], [100, 880], [460, 880], [0, 0], [560, 0], [0, 980], [560, 925]]), 'top_view')
    # draw foot
    foot = draw_points(changed, np.float32([[293, 960]]))
    cv2.imwrite('foot.png', foot)


__main__()
