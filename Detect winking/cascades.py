import cv2
import numpy as np
import argparse
import sys
import os
from os import listdir
from os.path import isfile, join

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-webcam', help="no path", action='store_true')
parser.add_argument('-video', help="requires path")
parser.add_argument('-image', help="requires  path")
parser.add_argument('-folder', help="requires path")
args = parser.parse_args()

###########################################################################

"""
Using cascade detectors
-----------------------
rectangles = cascade.detectMultiScale(
        gray_image, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)
Example:
--------
r = cascade.detectMultiScale(gray_image, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, (30,30))

rectangles: a list of rectangles. Each rectangle is: x, y, w, h
gray_image: the input image. Must be gray level, not color
scale factor: increasing/decreasing size by this factor
minNeighbors: number of neighbors that must also be detected.
         largest is 4, very selective in determining detection
	 small is 2,1, less selective in determining detection
	 0 - return all detections.
flag: 0 or  0|CV_HAAR_SCALE_IMAGE  
      0 scale template. 0|CV_HAAR_SCALE_IMAGE scale image.
      scaling template should be faster. scale image should be more accurate
minSize: size in pixels of smallest allowed detection

""" 

def detect_faces(cascade, gray_frame) :
    scaleFactor = 1.15 # range is from 1 to ..
    minNeighbors = 3   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (30,30) # range is from (0,0) to ..
    faces = cascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)
    return(faces)

def detect_eyes(cascade, gray_frame) :
    scaleFactor = 1.15 # range is from 1 to ..
    minNeighbors = 3   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (10,20) # range is from (0,0) to ..
    eyes = cascade.detectMultiScale(
        gray_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)
    return(eyes)
    

###########################################################################

def draw_boxes(frame, rectangles, color) :
    for rect in rectangles:
        draw_box(frame, rect, color)

def draw_box(frame, rect, color) :
    x, y, w, h = rect
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)


def detect_cascades(cascades, gray_frame) :
    face_cascade, eye_cascade = cascades

def detection_frame(cascades, frame) :
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    face_cascade, eye_cascade = cascades
    face_detections = detect_faces(face_cascade, gray_frame)
    face_color = (100,100,255)
    draw_boxes(frame, face_detections, face_color)
    eye_detections = detect_eyes(eye_cascade, gray_frame)
    eye_color = (100,255,100)
    draw_boxes(frame, eye_detections, eye_color)
    return

def detect_video(cascades, video_source) :
    windowName = "Video"
    showlive = True
    while(showlive):
        ret, frame = video_source.read()
        if not ret :
            showlive = False;
        else :
            detection_frame(cascades, frame)
            cv2.imshow(windowName, frame)
            if cv2.waitKey(30) >= 0:
                showlive = False
    # outside the while loop
    video_source.release()
    cv2.destroyAllWindows()
    return

###########################################################################

def runon_image(cascades, path) :
    frame = cv2.imread(path)
    detection_frame(cascades, frame)
    cv2.imshow("one image", frame)
    cv2.waitKey(0)
    return

def runon_webcam(cascades) :
    video_source = cv2.VideoCapture(0)
    if not video_source.isOpened():
        print("Can't open default video camera!")
        exit()
    detect_video(cascades, video_source)
    return

def runon_video(cascades, path) :
    video_source = cv2.VideoCapture(path)
    if not video_source.isOpened():
        print("Can't open video ", path)
        exit()
    detect_video(cascades, video_source)
    return

def runon_folder(cascades, path) :
    if(path[-1] != "/"):
        path = path + "/"
        files = [join(path,f) for f in listdir(path) if isfile(join(path,f))]
        for f in files:
            runon_image(cascades, f)
        return

if __name__ == "__main__":
    webcam = args.webcam
    video = args.video
    image = args.image
    folder = args.folder
    if not webcam and video is None and image is None and folder is None :
        print(
            "one argument from webcam,video,image,folder must be given.",
            "\n",
            "Example)",
            "-webcam or -video clip.avi or -image image.jpg or -folder images")
        sys.exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                         +'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')
    cascades = (face_cascade, eye_cascade)
    
    if webcam :
        runon_webcam(cascades)
    elif video is not None :
        runon_video(cascades,video)
    elif image is not None :
        runon_image(cascades,image)
    elif folder is not None :
        runon_folder(cascades,folder)
    else :
        print("impossible")
        sys.exit()

        cv2.destroyAllWindows()
