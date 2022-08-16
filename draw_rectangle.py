# import the necessary packages
import cv2
import os
import argparse
from glob import glob

# now let's initialize the list of reference point
ref_point = []
img_shape = None
image = None
window_name = None

def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, crop, window_name

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))

        initial_x = ref_point[0][0] / img_shape[1]
        initial_y = ref_point[0][1] / img_shape[0]
        width = (ref_point[1][0] - ref_point[0][0]) / img_shape[1]
        height = (ref_point[1][1] - ref_point[0][1]) / img_shape[0]
        print(initial_x, ' ', initial_y, ' ', width, ' ', height)


        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow(window_name, image)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())


def process_image(current_image):
    global image, img_shape, window_name

    window_name = current_image
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(current_image)
    img_shape = image.shape[:-1]
    clone = image.copy()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, shape_selection)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF

        # press 'r' to reset the window
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    # close all open windows
    cv2.destroyAllWindows()

if os.path.isdir(args["image"]):
    for file in sorted(glob(os.path.join(args["image"], "*hand*"))):
        process_image(file)
else:
    process_image(args["image"])


