import time
import numpy as np
import cv2
import threading
import socket
from math import atan2, degrees, sqrt
from random import sample

# Socket configuration
RASPBERRY_PI_IP = "192.168.1.100"  # Change to your Pi's IP
PORT = 5000
SOCKET_TIMEOUT = 2.0

# Initialize socket (global)
control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
control_socket.settimeout(SOCKET_TIMEOUT)

def connect_to_pi():
    """Establish connection to Raspberry Pi"""
    try:
        control_socket.connect((RASPBERRY_PI_IP, PORT))
        print("Connected to Raspberry Pi")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def send_command(command):
    """Send command to Raspberry Pi"""
    try:
        control_socket.sendall(command.encode() + b'\n')
        return True
    except Exception as e:
        print(f"Command failed: {e}")
        return False

# HSV ranges for blue (outer circle) and white (inner dot)
lower_blue = np.array([90, 100, 50])
upper_blue = np.array([130, 255, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# Target parameters
MIN_ARC_LENGTH = 30  # Minimum arc length in pixels to consider
MIN_CIRCLE_POINTS = 20  # Minimum points to attempt circle fitting
RANSAC_THRESHOLD = 3.0  # Pixel distance threshold for RANSAC inliers
MIN_INLIER_RATIO = 0.6  # Minimum inlier ratio for valid circle

video_capture = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video_capture.set(cv2.CAP_PROP_FPS, 30)

# Global variables for target information
target_found = False
target_center = (0, 0)
target_radius = 0
white_dot_found = False
white_dot_center = (0, 0)


def action_A(w_speed, y_speed):
    send_command(f"A {w_speed} {y_speed}")

def action_D(w_speed, y_speed):
    send_command(f"D {w_speed} {y_speed}")

def action_W(w_speed, y_speed):
    send_command(f"W {w_speed} {y_speed}")

def action_S():
    send_command("S")

def action_Stop():
    send_command("STOP")

# def action_A(w_speed, y_speed):
#     cmd.send_cmd(0x00, y_speed, w_speed, 0x00, 0x00)
#
#
# def action_D(w_speed, y_speed):
#     cmd.send_cmd(0x00, y_speed, w_speed, 0x00, 0x00)
#
#
# def action_W(w_speed, y_speed):
#     cmd.send_cmd(0x00, y_speed, w_speed, 0x00, 0x00)
#
#
# def action_S():
#     cmd.send_cmd(0x00, 0xca, 0x00, 0x00, 0x00)
#
#
# def action_Stop():
#     cmd.send_cmd(0x00, 0x00, 0x00, 0x00, 0x00)


def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


def fit_circle_ransac(points, iterations=100, threshold=RANSAC_THRESHOLD):
    """RANSAC-based circle fitting for partial circles"""
    if len(points) < 3:
        return None

    best_circle = None
    best_inliers = []

    for _ in range(iterations):
        # Randomly select 3 points
        sample_points = sample(points.tolist(), 3)

        try:
            # Fit circle to these 3 points
            (xc, yc), radius = cv2.minEnclosingCircle(np.array(sample_points))

            # Find inliers
            inliers = []
            for point in points:
                x, y = point[0]
                distance = abs(sqrt((x - xc) ** 2 + (y - yc) ** 2) - radius)
                if distance < threshold:
                    inliers.append(point)

            # Update best circle if we found more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_circle = ((xc, yc), radius)
        except:
            continue

    # Refine circle using all inliers
    if best_circle and len(best_inliers) >= MIN_CIRCLE_POINTS:
        (xc, yc), radius = best_circle
        inliers_array = np.array(best_inliers).reshape(-1, 2)
        (xc, yc), radius = cv2.minEnclosingCircle(inliers_array)
        return ((xc, yc), radius)

    return None


def find_target(frame):
    global target_found, target_center, target_radius, white_dot_found, white_dot_center

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold for blue color (outer circle)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    # Find contours in the blue mask
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_found = False
    white_dot_found = False
    target_center = (0, 0)
    target_radius = 0

    if len(contours) > 0:
        # Sort contours by length and process the longest ones first
        contours = sorted(contours, key=cv2.arcLength, reverse=True)

        for contour in contours:
            # Skip small contours
            if cv2.arcLength(contour, False) < MIN_ARC_LENGTH:
                continue

            # Try RANSAC circle fitting
            circle = fit_circle_ransac(contour)

            if circle:
                (center, radius) = circle
                center = (int(center[0]), int(center[1]))
                radius = int(radius)

                # Validate the circle
                if radius > 20:  # Minimum radius threshold
                    target_found = True
                    target_center = center
                    target_radius = radius

                    # Now look for the white dot inside the estimated circle area
                    mask_white = cv2.inRange(hsv, lower_white, upper_white)

                    # Create a circular ROI mask
                    target_mask = np.zeros_like(mask_white)
                    cv2.circle(target_mask, center, radius, 255, -1)

                    # Apply the mask to the white mask
                    masked_white = cv2.bitwise_and(mask_white, mask_white, mask=target_mask)

                    # Find contours in the white mask
                    white_contours, _ = cv2.findContours(masked_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(white_contours) > 0:
                        # Find the largest white contour (should be our dot)
                        largest_white = max(white_contours, key=cv2.contourArea)

                        # Get the center of the white dot
                        M = cv2.moments(largest_white)
                        if M['m00'] > 0:
                            white_dot_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                            # Only consider it a dot if it's within the target circle
                            distance_to_center = sqrt((white_dot_center[0] - center[0]) ** 2 +
                                                      (white_dot_center[1] - center[1]) ** 2)
                            if distance_to_center < radius * 0.8:
                                white_dot_found = True

                    break  # Found our target, stop processing other contours

    return target_found, white_dot_found


def process_frame(frame):
    global target_found, target_center, target_radius, white_dot_found, white_dot_center

    # Find the target and white dot
    find_target(frame)

    # Display information for debugging
    display_frame = frame.copy()
    if target_found:
        # Draw the estimated complete circle
        cv2.circle(display_frame, target_center, target_radius, (0, 255, 0), 2)
        cv2.circle(display_frame, target_center, 3, (0, 255, 255), -1)  # Center point

        if white_dot_found:
            # Draw the white dot center
            cv2.circle(display_frame, white_dot_center, 5, (0, 0, 255), -1)
            # Draw a line from target center to white dot
            cv2.line(display_frame, target_center, white_dot_center, (255, 0, 0), 2)

            # Calculate angle of the dot relative to the target center
            dx = white_dot_center[0] - target_center[0]
            dy = white_dot_center[1] - target_center[1]
            angle = degrees(atan2(dy, dx))
            cv2.putText(display_frame, f"Angle: {angle:.1f}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Detection', display_frame)

    # Control logic based on what we found
    T = None
    if target_found:
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        if white_dot_found:
            # We have both target and white dot - align to the dot
            offset_x = white_dot_center[0] - frame_center_x
            offset_y = white_dot_center[1] - frame_center_y

            # Map offsets to motor speeds
            w_speed = map_range(offset_x, -frame.shape[1] // 2, frame.shape[1] // 2, -100, 100)
            y_speed = map_range(offset_y, -frame.shape[0] // 2, frame.shape[0] // 2, -100, 100)

            # If we're close enough, stop
            if abs(offset_x) < 20 and abs(offset_y) < 20:
                T = threading.Thread(target=action_Stop)
            else:
                T = threading.Thread(target=action_W, args=(w_speed, y_speed))
        else:
            # We only see the target - center it
            offset_x = target_center[0] - frame_center_x
            w_speed = map_range(offset_x, -frame.shape[1] // 2, frame.shape[1] // 2, -100, 100)

            # Move forward slowly while centering
            T = threading.Thread(target=action_W, args=(w_speed, 30))
    else:
        # No target found - search for it
        T = threading.Thread(target=action_S)

    T.start()
    T.join()


def capture_frames():
    if not connect_to_pi():
        print("Failed to connect to Raspberry Pi")
        return

    try:
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                action_Stop()
                break

            ret, frame = video_capture.read()
            if ret:
                process_frame(frame)
    finally:
        control_socket.close()
        video_capture.release()
        cv2.destroyAllWindows()


# Start the camera thread
camera_thread = threading.Thread(target=capture_frames)
camera_thread.start()