from ultralytics import YOLO
import numpy as np
import cv2 as cv
import random


def main():
	# Initialize YOLO model
	model = YOLO("runs/detect/train7/weights/best.pt")



	# Initialize video capture (0 is the default camera)
	capture = cv.VideoCapture(0)
	while True:
		isTrue, frame = capture.read()

		frame = cv.resize(frame, (1920, 1080), interpolation=cv.INTER_LINEAR)

		# Perform object detection using YOLO
		results = model(frame)

		confs = results[0].boxes.conf
		bboxes = results[0].boxes.xyxy

		# Process and visualize detections
		for i in range(len(confs)):
			if confs[i] > 0.2:  # Adjust confidence threshold as needed
				# Extract bounding box coordinates
				x1, y1, x2, y2 = bboxes[i]

				roi = frame[int(y1):int(y2), int(x1):int(x2)]
				blured_roi = cv.blur(roi, (100, 100))
				
				frame[int(y1):int(y2), int(x1):int(x2)] = blured_roi
				
				
		cv.imshow("Video (press q to quit)", frame)

		if cv.waitKey(24) & 0xFF == ord('q'):
			break

	capture.release()
	cv.destroyAllWindows()





if __name__ == "__main__":
	main()

