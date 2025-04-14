import cv2
import numpy as np

# Define HSV range for blue exterior
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Define HSV range for white center
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# Initialize video capture
video_capture = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video_capture.set(cv2.CAP_PROP_FPS, 30)


def process_frame(frame):
    # Convert to HSV
    imgHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for blue exterior
    blue_mask = cv2.inRange(imgHSV, lower_blue, upper_blue)

    # Find contours for blue exterior
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask for white center
    white_mask = cv2.inRange(imgHSV, lower_white, upper_white)

    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 100:
            continue

        # Approximate contour to a circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Check if the contour is approximately circular
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:  # Adjust threshold for circularity
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find white center within the bounding box
        roi_white_mask = white_mask[y:y + h, x:x + w]
        white_contours, _ = cv2.findContours(roi_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if white_contours:
            # Assume the largest white contour is the center
            white_contour = max(white_contours, key=cv2.contourArea)
            M = cv2.moments(white_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + x
                cy = int(M['m01'] / M['m00']) + y
                # Highlight white center
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Display frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Blue Mask', blue_mask)
    cv2.imshow('White Mask', white_mask)


def capture_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_frames()