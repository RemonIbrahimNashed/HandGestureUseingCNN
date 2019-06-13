import cv2
import numpy as np
from time import sleep
import os

# global variables
bg = None

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

if __name__ == "__main__" :

		index = 0 

		aWeight = 0.5
		camera = cv2.VideoCapture(0)
		top, right, bottom, left = 10, 470, 250, 750
		num_frames = 0
		r = ""
		while True:
			
			(grabbed, frame) = camera.read()
			frame = cv2.flip(frame, 1)
			cv2.putText(frame,"predictio is "+r, (20,100), cv2.FONT_HERSHEY_PLAIN , 1.5, 100)
			clone = frame.copy()
			(height, width) = frame.shape[:2]
			roi = frame[top:bottom, right:left]
			gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (7, 7), 0)

			if num_frames < 30:
				run_avg(gray, aWeight)
			else:
				hand = segment(gray)
				if hand is not None:
					index +=1
					if index % 30 == 0  : 
						(thresholded, segmented) = hand
						thresholded = cv2.resize(thresholded, (64, 64))
						cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
						cv2.imshow("Thresholded", thresholded)
						sleep(3)
						path = "test"+str(index)+".jpg"
						cv2.imwrite(path,thresholded)
						r = os.popen("python predict.py "+path).read()[:-1]
						print("prediction is ",r)
						os.popen("rm -fr "+path).read()
						print("images taken: {}".format(index))

			cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
			num_frames += 1
			cv2.imshow("recording", clone)
			
			keypress = cv2.waitKey(1) & 0xFF
			if keypress == ord("q") :
				break
		
		cv2.destroyWindow("recording")
		cv2.destroyWindow("Thresholded")
		camera = None










































