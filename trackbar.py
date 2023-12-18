import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def filter_ice(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 30], dtype=np.uint8)
    upper = np.array([255, 130, 210], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    no_ice = cv2.bitwise_and(hsv, hsv, mask=np.invert(mask))
    return cv2.cvtColor(no_ice, cv2.COLOR_HSV2RGB)

def nothing(x):
    pass

file = "data/lots2.png"

img = cv2.imread(file)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.namedWindow('image')

cv2.createTrackbar('H','image',0,255,nothing)
cv2.createTrackbar('S','image',0,255,nothing)
cv2.createTrackbar('V','image',0,255,nothing)
cv2.createTrackbar('TOL','image',10,100,nothing)

to_show = img.copy()

while True:

    to_show = filter_ice(to_show)
    cv2.imshow("image", to_show)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    tol = cv2.getTrackbarPos('TOL', 'image')
    h = cv2.getTrackbarPos('H', 'image')
    s = cv2.getTrackbarPos('S', 'image')
    v = cv2.getTrackbarPos('V', 'image')

    hsv_val = np.array([h, s, v])
    top_range = np.clip(hsv_val + tol, 0, 255)
    bot_range = np.clip(hsv_val - tol, 0, 255)

    # bot_range[0] = 15; top_range[0] = 50
    bot_range[1] = 100; top_range[1] = 255
    bot_range[2] = 0; top_range[2] = 150
    #
    #
    mask = cv2.inRange(cv2.cvtColor(img.copy(), cv2.COLOR_RGB2HSV),
                         bot_range,
                         top_range)
    # to_show = mask.copy()
    to_show = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)
    print(f"Bottom: {bot_range} -- Top: {top_range} \n ")
    # time.sleep(0.1)

cv2.destroyAllWindows()
