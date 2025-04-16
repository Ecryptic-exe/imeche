import cv2
import argparse
import numpy as np


# import serial

def callback(value):
    pass


# ArduinoSerial=serial.Serial('com11',9600,timeout=0.1)

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range filter. RGB or HSV')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
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

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)
        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        x, y = 0, 0  # Initialize x and y

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask
            c = max(cnts, key=cv2.contourArea)

            # fit an ellipse to the contour if it has enough points
            if len(c) >= 5:
                ellipse = cv2.fitEllipse(c)
                center = (int(ellipse[0][0]), int(ellipse[0][1]))
                x, y = ellipse[0]

                # draw the ellipse
                cv2.ellipse(image, ellipse, (0, 255, 255), 2)

                # draw the center and information
                cv2.circle(image, center, 3, (0, 0, 255), -1)
                cv2.putText(image, "centroid", (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(image, f"({center[0]},{center[1]})", (center[0] + 10, center[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # Optionally draw the major and minor axes
                # (angle is in degrees, axes are half-lengths)
                angle = ellipse[2]
                axes = ellipse[1]
                major_axis = max(axes)
                minor_axis = min(axes)

                # Convert angle to radians for drawing lines
                angle_rad = np.deg2rad(angle)

                # Calculate endpoints for major axis
                x1 = int(center[0] + major_axis * np.cos(angle_rad))
                y1 = int(center[1] + major_axis * np.sin(angle_rad))
                x2 = int(center[0] - major_axis * np.cos(angle_rad))
                y2 = int(center[1] - major_axis * np.sin(angle_rad))

                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Calculate endpoints for minor axis (perpendicular to major axis)
                x3 = int(center[0] + minor_axis * np.cos(angle_rad + np.pi / 2))
                y3 = int(center[1] + minor_axis * np.sin(angle_rad + np.pi / 2))
                x4 = int(center[0] - minor_axis * np.cos(angle_rad + np.pi / 2))
                y4 = int(center[1] - minor_axis * np.sin(angle_rad + np.pi / 2))

                cv2.line(image, (x3, y3), (x4, y4), (0, 255, 0), 1)

        string = 'X{0:d}Y{1:d}'.format(int(x), int(y))
        print(string)
        # ArduinoSerial.write(string.encode('utf-8'))

        # show the frame to our screen
        cv2.imshow("Original", image)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()