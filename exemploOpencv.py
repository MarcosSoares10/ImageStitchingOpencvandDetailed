import numpy as np
import cv2
import imutils
from os import listdir

def resize_img(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img


def crop_image(img):
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
	# convert the stitched image to grayscale and set any pixel greater than 0 to 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find all external contours in the threshold image and select the largest contour which will be the contour/outline of  the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    # create two copies of the mask: one to serve as our actual
	# minimum rectangular region and another to serve as a counter
	# for how many pixels need to be removed to form the minimum
	# rectangular region
    minRect = mask.copy()
    sub = mask.copy()
	# keep looping until there are no non-zero pixels left in the
	# subtracted image
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
		# the thresholded image from the minimum rectangular mask
		# so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
	
    # find contours in the minimum rectangular mask and then
	# extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
	# use the bounding box coordinates to extract the our final
	# stitched image
    img = img[y:y + h, x:x + w]
    return img



def readImages(path,resize=False):
    imagesDirectory = listdir(path)
    list_path_images = []
    for i in imagesDirectory:
        list_path_images.append(path+i)

    list_path_images.sort()

    images_array = []
 
    for i in list_path_images:
        img = cv2.imread(i)
        if resize:
            img = resize_img(img,20)
        images_array.append(img) 
    
    return images_array


stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()


(status, stitched) = stitcher.stitch(readImages('imagens/',True)[0:2])

cropped = crop_image(stitched)
cv2.imwrite("Reconstructedbyopencv.png",cropped)



while(1):
        cv2.imshow("Image",cropped)
        k = cv2.waitKey(33)
        if k==27:   
            break
            cv2.destroyAllWindows()
        elif k==-1:  
            continue