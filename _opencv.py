import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# img =cv.imread('pics/cat.jpeg')
# cv.imshow('Cat',img)
#img_resized=rescaleImage(img,2)

'''
capture=cv.VideoCapture('swan.mp4')
0 is the webcam
1 is the second camera connected to computer and so on
to use an existing video-link

while True:
    #each frame is captured in frame, if captured isTrue is 1
    isTrue, frame=capture.read()
    frame_resized=rescaleImage(frame,0.2)
    cv.imshow('Video',frame)
    cv.imshow('Video resized',frame_resized)

    #the frame waits 20ms to go to next frame,if we press d it stops
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
'''

def rescaleImage(frame,scale=0.5):
    #will work for img,video,cam
    if frame is None:
        print("Error: Frame is None. Check your image path or video source!")
        return None 
    width=int(frame.shape[1]*scale)
    #frame.shape[1]=width of the img
    height=int(frame.shape[0]*scale)
    #frame.shape[0]=height of the img
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def ChangeRes(width,height):
    #call cv.VideoCapture to write this func
    #works on live video
    #in cv.VideoCapture each property has ID=>width=3, height=4
    #capture is a global variable=cv.VideoCapture() 
    capture=cv.VideoCapture(0)#defining inside the func for simplicity
    capture.set(3,width)
    capture.set(4,height)

blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow('Blank',blank)

#TO SPECIFY A RECTANGLE in a pic
# blank[200:300,300:400]=0,0,255
# cv.imshow('Green',blank)

#Draw rectangle
#we can specify thickness,else it takes its default
#if thickness =-1 or cv.FILLED to fill the rectangle
#width is=>img.shape[0], height=img.shape[1]
#in the below example it fills the color (0,255,0) from (0,0) to (width/2,height/2)
cv.rectangle(blank,(0,0),(blank.shape[0]//2,blank.shape[1]//2),(0,255,0),thickness=cv.FILLED)
cv.imshow("rectangle",blank)
# cv.waitKey(0)

#Draw Circle
#img, centre,radius,color,thickness
cv.circle(blank,(250,250),40,(0,255,255),thickness=-1)
cv.imshow("Circle",blank)
# cv.waitKey(0)

#Draw line
#img,one end coordinates,2nd end coordinates,color,thickness
cv.line(blank,(0,0),(250,250),(0,255,0),thickness=2)
cv.imshow("LIne",blank)

#Write Text on an image
#img,text,point,handwriting,scale,color,thickness
cv.putText(blank,'hello',(225,225),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
cv.imshow("text",blank)
cv.waitKey(0)

#Apply filters on image
'''
*GRAYSCALE IMAGE
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

*BLUR-GaussianBlur
second argument=>ksize=(width,height), more value of h/width=>more blur
blur=cv.GaussianBlur(img,(9,9),cv.BORDER_DEFAULT)

*EDGE CASCADE-to get the image edges
canny=cv.Canny(img,125,175)
//blur images have less edges

*DILATED
ksize=(3,3)
dilated=cv.dilate(img,(3,3),iterations=1)

*ERODED
eroded=cv.erode(dilated,(3,3),iterations=1)

*RESIZE
interpolation argument:
interpolation=cv.INTER_AREA(useful when we want to shrink the img)
interpolation=cv.INTER_LINEAR/cv.INTER_CUBIC(useful when we want to enlarge the image)
resized=cv.resize(img,(500,500),interpolation=)
//ignores the aspect ratio

*CROP
//similar to array slicing
cropped=img[50:150,50,150]
'''
#IMAGE TRANSFORMATIONS
#TRANSLATION
def translate(img,x,y):
    transMat=np.float32([[1,0,x],[0,1,y]])
    dimen=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimen)
#x=>right
#y=>down

#ROTATE
#angle in degrees
def rotate(img,angle,rotPoint=None):
    (height,width)=img.shape[:2]
    if rotPoint is None:
        rotPoint=(width//2,height//2)
    #the below take scale as the last argument
    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)
    return cv.warpAffine(img,rotMat,dimensions)

#RESIZE
#resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)

#FLIP
# flip=cv.flip(img,-1)
#flipCode=0/1/-1
# 0->vertical flipping(x-axis)/up<->down
# 1->horicontal flipping(y-axis)/left<->right
# -1->both horizontal and vertical

#FIND THE CONTOUR LIST
'''
contours,hierachies=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
contours is a list of the all coordinates of the edges
#contour return parameter
cv.RETR_LIST returns all the contours
cv.RETR_EXTRNAL returns external contours
cv.RETR_TREE returns hierarchial contours
#contour approximation method
cv.CHAIN_APPROX_NONE=>does nothing returns all contours
cv.CHAIN_APPROX_SIMPLE=>gives only compresses contour coordinates
print(len(contours),"contours found")
'''

#BINARIZE IMAGES
'''
#thresh used to binarize the image(use only two colours)
ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
#if the intensity >125 it is set to 255,else it is set to 0(black)
cv.imshow('thresh',thresh)
'''

#DRAW CONTOURS ON ANOtHER IMAGE
'''
ctrs,hierachies=cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
#drawContours to draw the contours on a given image
#parameters: image,contour list,contour idx,color(bgr),thickness
cv.drawContours(blank,ctrs,-1,(0,0,255),2)
'''



#ADVANCED:
#COLOR SPACES
'''
#BGR TO HSV 
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)

#BGR TO LAB(L*A*B)
lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)

#BGR TO RGB
rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)

#BGR TO gray
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

=>cannot ocnvert gray to hsv directly,gray=>bgr=>hsv
'''

#COLOR CHANNELS
'''
b,g,r=cv.split(img)=>b, g,r represents grayscale images 
     =>each image is the intensity of that color in the original img
     =>they will be 1D array of intensities
merged=cv.merge([b,g,r]) =>merges three channels to give original image
'''

#BLURRING
'''
#Averaging
#each pixel is make equal to the average of all the surrounding pixels
average=cv.blur(img,(3,3))
=>ksize(3,3) increase to increase blur effect

#GAUSSIAN BLUR
blur=cv.GaussianBlur(img,(7,7),0)
=>thirs parameter is sigmaX or the standard deviation

#MEDIAN BLUR
#same as average but median is used instead of average
blur=cv.medianBlur(img,3)
=>ksize is given as a single integer

#Bilateral
blur=cv.bilateralFilter(img,5,15,15)
=>the values determine till which pixel distance determine the current one
=>if its more,the pixels even farther away affect the current one
'''

#BITWISE OPERATORS
'''
#bitwise AND-intersecting regions
bw_AND=cv.bitwise_and(rectangle,circle)

#bitwise_OR-intersecting and non intersecting regions
bw_OR=cv.bitwise_or(rectangle,circle)

#bitwise XOR-non-intersecting regions
bw_XOR=cv.bitwise_xor(rectangle,circle)

#bitwise NOT-white<->black
bw_NOT=cv.bitwise_not(rectangle)
'''

#MASKING
'''
size of masking image shud be same as the original
'''

#HISTOGRAMS
'''
gray_hist=cv.calcHist([gray],[0],None,[256],[0,256])


'''

#THRESHHOLDING
'''
#Simple threshold
#if intensity > second arg,it sets to white
threshold,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('Simple threshhold',thresh)

#inverse
threshold,thresh_inv=cv.threshold(gray,125,255,cv.THRESH_BINARY_INV)
cv.imshow('Simple threshhold Inverse',thresh_inv)

#adaptive thresh
adaptive_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,3)
cv.imshow('adaptive thresh',adaptive_thresh)
'''

#FILTERS
'''
#Laplacian
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)

#Sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combined=cv.bitwise_or(sobelx,sobely)
cv.imshow('sobelx',sobelx)
cv.imshow('sobely',sobely)
cv.imshow('combined',combined)

'''






#-215 error: could not find image/could not read
# waits for a keyboard key to be presses to close(useful for images)
cv.waitKey(0)



'''
IDs for VideoCapture properties
1	CAP_PROP_POS_MSEC	Current position of the video (in milliseconds).
2	CAP_PROP_POS_FRAMES	Current frame index.
3	CAP_PROP_FRAME_WIDTH	Frame width (in pixels).
4	CAP_PROP_FRAME_HEIGHT	Frame height (in pixels).
5	CAP_PROP_FPS	Frames per second (FPS).
6	CAP_PROP_FOURCC	4-character codec (e.g., MJPG).
7	CAP_PROP_FRAME_COUNT	Total number of frames.
10	CAP_PROP_BRIGHTNESS	Brightness level (only for cameras).
11	CAP_PROP_CONTRAST	Contrast level (only for cameras).
12	CAP_PROP_SATURATION	Saturation level (only for cameras).

'''