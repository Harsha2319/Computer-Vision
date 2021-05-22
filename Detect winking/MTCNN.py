import cv2
import numpy as np
import argparse
import sys
import os
from os import listdir
from os.path import isfile, join

from mtcnn_cv2 import MTCNN

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-webcam', help="no path", action='store_true')
parser.add_argument('-video', help="requires path")
parser.add_argument('-image', help="requires  path")
parser.add_argument('-folder', help="requires path")
args = parser.parse_args()


###########################################################################

def detect_eyes(eye_cascade, gray_frame):
    scaleFactor = 1.12  # range is from 1 to ..
    minNeighbors = 5  # range is from 0 to ..
    flag = 0 | cv2.CASCADE_SCALE_IMAGE  # either 0 or 0|cv2.CASCADE_SCALE_IMAGE
    minSize = (10, 20)  # range is from (0,0) to ..
    eyes = eye_cascade.detectMultiScale(
        gray_frame,
        scaleFactor,
        minNeighbors,
        flag,
        minSize)
    return (eyes)

def draw_eye_boxes(frame, rectangles, color) :
    for rect in rectangles:
        draw_eye_box(frame, rect, color)

def draw_eye_box(frame, rect, color) :
    x, y, w, h = rect
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

###########################################################################


def drawbox(frame, box, boxcolor, confidence):
    # print(f)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    cv2.rectangle(frame, (x, y), (x + w, y + h), boxcolor, 2)
    textcolor = (255, 155, 0)
    text = "{:.4f}".format(confidence)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textcolor, 2)
    return (frame)


def draw_boxes(frame, eye_detections, detections):
    number_detections = 0
    detection_color = (0, 155, 255)
    non_detection_color = (155, 255, 0)

    if len(detections) > 0:
        for f in detections:
            box = f['box']
            confidence = f['confidence']
            if (wink(frame, eye_detections, f)):
                number_detections += 1
                drawbox(frame, box, detection_color, confidence)
            #else:
                #drawbox(frame, box, non_detection_color, confidence)
    return (number_detections)


def wink(frame, eye_detections, f):
    count = 0
    box = f['box']
    confidence = f['confidence']
    keypoints = f['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']

    #cv2.rectangle(frame, (left_eye[0], left_eye[1]), (left_eye[0] + 5, left_eye[1] + 5), (0, 0, 0), -1)
    #cv2.rectangle(frame, (right_eye[0], right_eye[1]), (right_eye[0] + 5, right_eye[1] + 5), (0, 0, 0), -1)
    #cv2.rectangle(frame, (nose[0], nose[1]), (nose[0] + 5, nose[1] + 5), (0, 0, 0), -1)
    #cv2.rectangle(frame, (mouth_left[0], mouth_left[1]), (mouth_left[0] + 5, mouth_left[1] + 5), (0, 0, 0), -1)
    #cv2.rectangle(frame, (mouth_right[0], mouth_right[1]), (mouth_right[0] + 5, mouth_right[1] + 5), (0, 0, 0), -1)

    eye_c = []

    for eye in eye_detections:
        if eye[0] >= box[0] and eye[0] < box[0]+box[2] and eye[1] > box[1] and eye[1] < box[1]+int(box[3]/2):
            count += 1
            eye_c.append(eye)

    """
    if count == 0:
        d1 = int((mouth_left[1] - left_eye[1]) / 3)
        d2 = int((mouth_right[1] - right_eye[1]) / 3)
        print(mouth_left[1], left_eye[1], (mouth_left[1] - left_eye[1]), ((mouth_left[1] - left_eye[1])/6), d1, left_eye[0]-d1, left_eye[1]-d1)
        h_eye1 = [left_eye[0]-d1, left_eye[1]-d1, 2*d1, 2*d1]
        h_eye2 = [right_eye[0]-d2, right_eye[1]-d2, 2*d2, 2*d2]
        cv2.rectangle(frame, (h_eye1[0], h_eye1[1]), (h_eye1[0]+h_eye1[2], h_eye1[1]+h_eye1[3]), (0, 0, 0), 2)
        cv2.rectangle(frame, (h_eye2[0], h_eye2[1]), (h_eye2[0]+h_eye2[2], h_eye2[1]+h_eye2[3]), (0, 0, 0), 2)
        eye_c = [h_eye1, h_eye2]
        count = 2
    """

    if count == 1:
        return True

    if count == 2:
        image1 = frame[eye_c[0][0] : eye_c[0][0]+eye_c[0][2], eye_c[0][1] : eye_c[0][1]+eye_c[0][3]]
        image2 = frame[eye_c[1][0] : eye_c[1][0]+eye_c[1][2], eye_c[1][1] : eye_c[1][1]+eye_c[1][3]]
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        histogram1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        histogram2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

        c1 = 0

        # Euclidean Distace between data1 and test
        i = 0
        while i < 256:
            c1 += (histogram1[i] - histogram2[i])**2
            i += 1
        c1 = c1 ** (1 / 2)

        if c1 < 400:
            return True
        return False


def detection_frame(eye_cascade, detector, frame):

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    eye_detections = detect_eyes(eye_cascade, gray_frame)
    eye_color = (100, 255, 100)
    #draw_eye_boxes(frame, eye_detections, eye_color)

    detections = detector.detect_faces(frame)
    number_detections = draw_boxes(frame, eye_detections, detections)
    return (number_detections)


def detect_video(eye_cascade, detector, video_source):
    windowName = "Video"
    showlive = True
    while (showlive):
        ret, frame = video_source.read()
        if not ret:
            showlive = False;
        else:
            detection_frame(eye_cascade,detector, frame)
            cv2.imshow(windowName, frame)
            if cv2.waitKey(30) >= 0:
                showlive = False
    # outside the while loop
    video_source.release()
    cv2.destroyAllWindows()
    return


###########################################################################

def runon_image(eye_cascade, detector, path):
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections_in_frame = detection_frame(eye_cascade, detector, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("one image", frame)
    cv2.waitKey(0)
    return detections_in_frame


def runon_webcam(eye_cascade, detector):
    video_source = cv2.VideoCapture(0)
    if not video_source.isOpened():
        print("Can't open default video camera!")
        exit()

    detect_video(eye_cascade, detector, video_source)
    return


def runon_video(eye_cascade, detector, path):
    video_source = cv2.VideoCapture(path)
    if not video_source.isOpened():
        print("Can't open video ", path)
        exit()
    detect_video(eye_cascade, detector, video_source)
    return


def runon_folder(eye_cascade, detector, path):
    if (path[-1] != "/"):
        path = path + "/"
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    all_detections = 0
    for f in files:
        f_detections = runon_image(eye_cascade, detector, f)
        all_detections += f_detections
    return all_detections


if __name__ == '__main__':
    webcam = args.webcam
    video = args.video
    image = args.image
    folder = args.folder
    if not webcam and video is None and image is None and folder is None:
        print(
            "one argument from webcam,video,image,folder must be given.",
            "\n",
            "Example)",
            "-webcam or -video clip.avi or -image image.jpg or -folder images")
        sys.exit()

    detector = MTCNN()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if webcam:
        runon_webcam(eye_cascade, detector)
    elif video is not None:
        runon_video(eye_cascade, detector, video)
    elif image is not None:
        runon_image(eye_cascade, detector, image)
    elif folder is not None:
        all_detections = runon_folder(eye_cascade, detector, folder)
        print("total of ", all_detections, " detections")
    else:
        print("impossible")
        sys.exit()

    cv2.destroyAllWindows()


