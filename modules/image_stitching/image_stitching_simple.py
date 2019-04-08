# USAGE
# python image_stitching_simple.py --images images/scottsdale --output output.png

# import the necessary packages
import cv2

# construct the argument parser and parse the arguments


#
# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
image1 = cv2.imread('test_image_1.jpg')
image2 = cv2.imread('test_image_2.jpg')
image3 = cv2.imread('test_image_3.jpg')
images = [image1, image2, image3]

# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
    # write the output stitched image to disk
    cv2.imwrite('output_stitching_simple.jpg', stitched)

    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed ({})".format(status))
