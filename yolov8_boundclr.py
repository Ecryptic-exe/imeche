import cv2
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from yolo_segmentation import YOLO_Segmentation
from yolo_segmentation import YOLO_Detection
import time
import numpy as np

CLASSES = yaml_load(check_yaml('data.yaml'))['names']
print(CLASSES)
# ys = YOLO_Segmentation("yolov8m-seg.pt")
yd = YOLO_Detection("best.pt")

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, img = cam.read()
    startTime = time.time()
    # bboxes, classes, segmentations, scores = ys.detect(img)
    bboxes, class_ids, scores = yd.detect(img)
    # for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        (x, y, x2, y2) = bbox
        if score > 0.6:
            # Extract the region within the bounding box
            roi = img[y:y2, x:x2]

            # Convert the ROI to HSV color space
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define the range for white color in HSV
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])

            # Create a mask for the white color
            mask = cv2.inRange(hsv_roi, lower_white, upper_white)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Calculate the centroid of the largest contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw the centroid on the image
                    cv2.circle(roi, (cx, cy), 5, (0, 255, 0), -1)

                    # Draw the bounding box
                    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

                    # Display the label
                    label = f'{CLASSES[class_id]} ({score:.2f})'
                    cv2.putText(img, label, (x, y - 10), font, 2, (0, 0, 255), 2)

    newTime = time.time()
    FPS = str(int(1 / (newTime - startTime)))
    cv2.putText(img, FPS, (20, 50), font, 3, (255, 0, 0), 3)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xff == 27:
        break
cam.release()
cv2.destroyAllWindows()