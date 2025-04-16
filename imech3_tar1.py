import cv2
import numpy as np

# Define color ranges in HSV
COLOR_RANGES = {
    'yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([30, 255, 255]),
        'color': (0, 255, 255)  # Yellow in BGR
    },
    'red': {
        'lower': np.array([0, 150, 50]),
        'upper': np.array([10, 255, 255]),
        'color': (0, 0, 255)  # Red in BGR
    },
    'blue': {
        'lower': np.array([71, 56, 36]),
        'upper': np.array([180, 220, 220]),
        'color': (255, 0, 0)  # Blue in BGR
    }
}


def detect_sphere(frame, color_name):
    """Detect a sphere of specified color"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_data = COLOR_RANGES[color_name]

    mask = cv2.inRange(hsv, color_data['lower'], color_data['upper'])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            return {
                'ellipse': ellipse,
                'center': center,
                'color': color_name,
                'color_bgr': color_data['color']
            }
    return None


def get_movement_commands(center, frame_width, frame_height, tolerance=20):
    """Determine movement commands based on centroid position"""
    center_x, center_y = frame_width // 2, frame_height // 2
    x, y = center

    # X-axis movement
    if abs(x - center_x) <= tolerance:
        x_cmd = "CENTER"
    elif x < center_x - tolerance:
        x_cmd = "GO LEFT"
    else:
        x_cmd = "GO RIGHT"

    # Y-axis movement
    if abs(y - center_y) <= tolerance:
        y_cmd = "CENTER"
    elif y < center_y - tolerance:
        y_cmd = "GO STRAIGHT"
    else:
        y_cmd = "GO BACK"

    return x_cmd, y_cmd


def main():
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw center lines and crosshair
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

        # Try detecting spheres in order of priority
        detected = None
        for color in ['yellow', 'red', 'blue']:
            detected = detect_sphere(frame, color)
            if detected:
                break

        if detected:
            # Draw the detected sphere
            cv2.ellipse(frame, detected['ellipse'], detected['color_bgr'], 3)
            cv2.circle(frame, detected['center'], 5, detected['color_bgr'], -1)

            # Display sphere info
            cv2.putText(frame, detected['color'].upper(),
                        (detected['center'][0] + 15, detected['center'][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        detected['color_bgr'], 2)

            cv2.putText(frame, f"({detected['center'][0]},{detected['center'][1]})",
                        (detected['center'][0] + 15, detected['center'][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        detected['color_bgr'], 1)

            # Get movement commands
            x_cmd, y_cmd = get_movement_commands(detected['center'], width, height)

            # Display movement commands
            cv2.putText(frame, f"X: {x_cmd}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Y: {y_cmd}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display detection info
            cv2.putText(frame, f"Detected: {detected['color']}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No sphere detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display instructions
        cv2.putText(frame, "Detection priority: Yellow -> Red -> Blue", (10,  height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(frame, "Press 'q' to quit", (10, height - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Smart Sphere Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()