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

if not os.path.exists("Dataset"):
	os.mkdir("Dataset")
if not os.path.exists("Dataset/training_set"):
	os.mkdir("Dataset/training_set")

if not os.path.exists("Dataset/test_set"):
	os.mkdir("Dataset/test_set")

dirs=["1_one/1_one",'2_two/2_two','3_three/3_three','4_four/4_four', '5_five/5_five']
sets={'training_set':4000,'test_set':800}

for set_name in sets:
	print("Taking images for the {}. Press enter when ready. ".format(set_name.upper()))
	raw_input()
	if not os.path.exists("Dataset"):os.mkdir("Dataset/{}".format(set_name))
	for dir_name in dirs:
		print("""\nTaking images for the {} dataset. Press enter whenever ready. Note: Place the gesture to be recorded inside the green rectangle shown in the preview until it automatically disappears.""".format(dir_name))
		raw_input()
		for _ in range(5):
			print(5-_)
			sleep(1)
		print("GO!")
		if not os.path.exists("Dataset/{}/{}".format(set_name,os.path.basename(dir_name))):
			os.mkdir("Dataset/{}/{}".format(set_name,os.path.basename(dir_name)))
		path = "./Dataset/{}/{}/".format(set_name,dir_name.split("/")[0])
		index = os.popen("ls "+path+" | wc -l ").read()[:-1]
		index = int(index)
		index +=1
		end = index + sets[set_name]

		aWeight = 0.5
		camera = cv2.VideoCapture(0)
		top, right, bottom, left = 10, 470, 250, 750
		num_frames = 0
		while True:
			
			(grabbed, frame) = camera.read()
			frame = cv2.flip(frame, 1)
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
					(thresholded, segmented) = hand
					thresholded = cv2.resize(thresholded, (64, 64))
					cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
					cv2.imshow("Thresholded", thresholded)
					cv2.imwrite("Dataset/{}/".format(set_name)+str(dir_name)+"{}.jpg".format(index),thresholded)
					print("images taken: {}".format(index))

			cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
			num_frames += 1
			cv2.imshow("recording", clone)

			keypress = cv2.waitKey(1) & 0xFF
			if keypress == ord("q") or index==end:
				break
		
		cv2.destroyWindow("recording")
		cv2.destroyWindow("Thresholded")
		camera = None










































