import cv2
import numpy as np
from matplotlib import pyplot as plt


def affine_transform():
    # 1. Find affine transform ( cv2.getAffineTransform() )
    # 2. Use the undocumented set of parameters for performing the warp:
    # res = cv2.warpAffine(src_img, AffineTransfMat,(dst_width, dst_height),dst_img,borderMode=cv2.BORDER_TRANSPARENT)
    # 3. For the best looking result, the aspect ratio of the square surface and the source imageshould match.

    img = cv2.imread('test.jpg')
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols, rows))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

    pass


def projective_transform(img):
    pass


def __main__():
    affine_transform()


__main__()
