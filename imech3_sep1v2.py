import cv2
import argparse
import numpy as np


def callback(value):
    pass


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)
    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255
        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True, help='Range filter. RGB or HSV')
    ap.add_argument('-w', '--webcam', required=False, help='Use webcam', action='store_true')
    args = vars(ap.parse_args())
    if not args['filter'].upper() in ['RGB', 'HSV']:
        ap.error("Please specify a correct filter.")
    return args


def get_trackbar_values(range_filter):
    values = []
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)
    return values


def main():
    args = get_arguments()
    range_filter = args['filter'].upper()
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    setup_trackbars(range_filter)

    while True:
        if args['webcam']:
            ret, image = camera.read()
            if not ret:
                break

            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Get frame dimensions
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        tolerance = 20

        # Draw center lines for reference
        cv2.line(image, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.line(image, (0, center_y), (width, center_y), (255, 255, 255), 1)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        x, y = center_x, center_y  # Default to center if no object detected

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if len(c) >= 5:
                ellipse = cv2.fitEllipse(c)
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                x, y = ellipse[0]

                cv2.ellipse(image, ellipse, (0, 255, 255), 2)
                cv2.circle(image, center, 3, (0, 0, 255), -1)

        # Determine X direction
        if abs(x - center_x) <= tolerance:
            x_direction = "CENTER"
        elif x < center_x - tolerance:
            x_direction = "GO LEFT"
        else:
            x_direction = "GO RIGHT"

        # Determine Y direction
        if abs(y - center_y) <= tolerance:
            y_direction = "CENTER"
        elif y < center_y - tolerance:
            y_direction = "GO STRAIGHT"
        else:
            y_direction = "GO BACK"

        # Display directions
        cv2.putText(image, f"X: {x_direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Y: {y_direction}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Position: ({x},{y})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Original", image)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()