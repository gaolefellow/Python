import cv2 as cv
import numpy as np
img = cv.imread('./pic/5.png',0)

l,w = img.shape
print(l,w)

l_mid = l//2
w_mid = w//2
short = min(l,w)

img_sqart = img[0:l,w_mid-short//2:w_mid+short//2]

kernel = np.ones((10, 10), np.uint8)
erosion = cv.erode(img_sqart, kernel, iterations=5)

img_resize = cv.resize(erosion,(28,28),cv.INTER_LINEAR)

black = np.ones((28,28))*255
img_resize = black-img_resize
print(img_resize)

cv.imwrite('./pic/5_1.png',img_resize)
