import cv2
import numpy as np
import imutils

img = cv2.imread('./data/video/cafetefria_ground.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = np.array([0,210,255]) # hmin smin vmin
upper_range = np.array([179,255,255]) # hmax smax vmax

mask = cv2.inRange(hsv,lower_range,upper_range)
kernel = np.ones((2, 2))
img_erode = cv2.erode(mask, kernel, iterations=1)

final_pic = cv2.bitwise_and(img,img,mask=img_erode)



cv2.imshow("image",img)
cv2.imshow('Mask',mask)
cv2.imshow('final',img_erode)
cv2.imshow('final2',final_pic)

# cv2.imwrite('./data/video/cafeteria_mask.png',img_erode)



cv2.waitKey()

cv2.destroyAllWindows()




## if check contour
# import cv2
# import numpy as np
# import imutils
#
# img = cv2.imread('./cafetefria_ground.png')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# lower_range = np.array([0,210,255]) # hmin smin vmin
# upper_range = np.array([179,255,255]) # hmax smax vmax
#
# mask = cv2.inRange(hsv,lower_range,upper_range)
#
# #contours
# cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
#
#
# # cnts1 = cv2.findContours(thresh1.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts1 = imutils.grab_contours(cnts1)
#
#
# for c in cnts:
#     cv2.drawContours(img,[c],-1,(0,0,255),2)
#
# cv2.imshow("image",img)
# cv2.imshow('Mask',mask)
# cv2.waitKey()
#
# cv2.destroyAllWindows()