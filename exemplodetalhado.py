import cv2
import numpy as np
from os import listdir
import imutils

def resize_img(img,scale_percent=20):
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

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    elif method == 'flann':
        
        FLANN_INDEX_LINEAR = 0
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_KMEANS = 2
        FLANN_INDEX_COMPOSITE = 3
        FLANN_INDEX_KDTREE_SINGLE = 4
        FLANN_INDEX_HIERARCHICAL = 5
        FLANN_INDEX_LSH = 6
        FLANN_INDEX_SAVED = 254
        FLANN_INDEX_AUTOTUNED = 255
        
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    return matcher

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, _) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H)
    else:
        return None

def constructImage(source_image,target_image, M):
    #Taking the Homography generated previously
    (matches, H) = M

    #Taking the height and width of source and target image
    height_src_img,width_src_img = source_image.shape[:2]
    height_tgt_img,width_tgt_img = target_image.shape[:2]

    #Taking the corners of the images: top-left, bottom-left, bottom-right, top-right
    array_corners_source_img = np.float32([[0,0],[0,height_src_img],[width_src_img,height_src_img],[width_src_img,0]]).reshape(-1,1,2)
    array_corners_target_img = np.float32([[0,0],[0,height_tgt_img],[width_tgt_img,height_tgt_img],[width_tgt_img,0]]).reshape(-1,1,2)
    


    #Applying perspectiveTransform to conners of source_image
    array_corners_source_img_ = cv2.perspectiveTransform(array_corners_source_img, H)
    full_img_array_corners = np.concatenate((array_corners_source_img_, array_corners_target_img), axis=0)

    #Taking max min of x,y corners coordinate
    [xmin, ymin] = np.int64(full_img_array_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int64(full_img_array_corners.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]


    width = width_src_img + width_tgt_img
    height = height_src_img + height_tgt_img
    
    #Translation 
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    
    source_image_warped = cv2.warpPerspective(source_image, Ht.dot(H),  (width, height))    
    source_image_warped[t[1]:height_tgt_img+t[1],t[0]:width_tgt_img+t[0]] = target_image
   

    return source_image_warped.astype('uint8')

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    matcher = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    if method == 'flann':
        featuresA =  np.float32(featuresA)
        featuresB =  np.float32(featuresB)
    
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def detectAndDescribe(image, method):
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create() #Deprecated/removed on opencv 4
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    kps, features = descriptor.detectAndCompute(image, None)
    
    return kps, features

def preProcessimage(img, feature_extractor):
    #preProcessing and return features and keypoints
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       
    edge = cv2.Canny(img,100,255)
    kpsA, featuresA = detectAndDescribe(img, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(edge, method=feature_extractor)
    features = np.concatenate((featuresA,featuresB))
    keypoints = np.concatenate((kpsA,kpsB))
    return keypoints, features

def executeProcess(referenceimage,nextimage, feature_extractor, matcher_method):

    keypointsprevious,featuresprevious = preProcessimage(referenceimage,feature_extractor)
        
    keypointsnext,featuresnext = preProcessimage(nextimage,feature_extractor)
    
    matches = matchKeyPointsKNN(featuresprevious, featuresnext, ratio=0.85, method=matcher_method)
    

    M = getHomography(keypointsprevious, keypointsnext, featuresprevious,featuresnext, matches, reprojThresh=5)
    if M is None:
        print("Error!")

    recontructed_img = constructImage(referenceimage, nextimage, M)
    return recontructed_img

def manageExecution(images_array,feature_extractor, matcher_method):
    for i in range(0,len(images_array)-1):
        print(i)
        if i==0:
            imgA = images_array[i]
            imgB = images_array[i+1]
        else:
            imgB = images_array[i+1]
        imgA = executeProcess(imgA,imgB,feature_extractor, matcher_method)
        imgA = crop_image(imgA)
        

    return imgA

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
            img = resize_img(img,scale_percent=20)
        images_array.append(img) 
    
    return images_array

feature_extractor = 'sift'
matcher_method = 'flann'
path = 'imagens/'
fullimage = manageExecution(readImages(path,True)[0:5],feature_extractor, matcher_method)
cv2.imwrite("panoramicimage.png",fullimage)

while(1):
        cv2.imshow("Panorama",fullimage)
        k = cv2.waitKey(33)
        if k==27:   # ESC to exit
            break
            cv2.destroyAllWindows()
        elif k==-1:  
            continue