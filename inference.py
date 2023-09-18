from ultralytics import YOLO
import numpy as np
import cv2 as cv
import random


def main():
	# Initialize YOLO model
	model = YOLO("runs/detect/train7/weights/best.pt")

	color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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
			if confs[i] > 0.5:  # Adjust confidence threshold as needed
				# Extract bounding box coordinates
				x1, y1, x2, y2 = bboxes[i]

				thickness = 2
				cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
							 color, thickness)

		cv.imshow("Video (press q to quit)", frame)

		if cv.waitKey(24) & 0xFF == ord('q'):
			break

	capture.release()
	cv.destroyAllWindows()





if __name__ == "__main__":
	main()

