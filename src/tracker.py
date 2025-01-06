import cv2
import numpy as np
from picamera2 import Picamera2

def points_to_tennis_score(points):
    """Convierte la cantidad de puntos (0,1,2,3,4,...) a la nomenclatura básica del tenis."""
    if points == 0:
        return "Love"
    elif points == 1:
        return "15"
    elif points == 2:
        return "30"
    elif points == 3:
        return "40"
    else:
        return "Game"

def init_kalman_filter():
    """Inicializa un filtro de Kalman sencillo para estimar estado y medidas."""
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman

def process_stream(picamera, seconds_absence=1):
    """Procesa el stream en tiempo real usando Picamera2."""
    fps = 30  # Estimamos 30 FPS para la Picamera2
    absence_frames_threshold = int(seconds_absence * fps)

    lower_green = np.array([29, 86, 6], dtype="uint8")
    upper_green = np.array([64, 255, 255], dtype="uint8")

    points = 0
    games = 0
    absent_frames = 0
    already_counted = False

    kalman = init_kalman_filter()
    frame_shape = picamera.capture_array().shape
    height, width, _ = frame_shape
    kalman.statePost = np.array([width // 2, height // 2, 0, 0], dtype=np.float32)

    last_w, last_h = 60, 60

    try:
        while True:
            frame = picamera.capture_array()

            predicted = kalman.predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ball_in_current_frame = False
            detected_center = None

            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)

                if radius > 10:
                    M = cv2.moments(c)
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 2)

                    ball_in_current_frame = True
                    detected_center = (center_x, center_y)
                    last_w, last_h = w_rect, h_rect

            if ball_in_current_frame and detected_center is not None:
                measurement = np.array([[np.float32(detected_center[0])], [np.float32(detected_center[1])]])
                kalman.correct(measurement)
                ref_x = int(kalman.statePost[0])
                ref_y = int(kalman.statePost[1])
            else:
                ref_x, ref_y = pred_x, pred_y

            cv2.circle(frame, (ref_x, ref_y), 5, (255, 0, 0), -1)
            x_bb = int(ref_x - last_w // 2)
            y_bb = int(ref_y - last_h // 2)
            cv2.rectangle(frame, (x_bb, y_bb), (x_bb + last_w, y_bb + last_h), (255, 0, 255), 2)

            if ball_in_current_frame:
                absent_frames = 0
                already_counted = False
            else:
                absent_frames += 1
                if (not already_counted) and (absent_frames >= absence_frames_threshold):
                    points += 1
                    already_counted = True
                    if points >= 4:
                        games += 1
                        points = 0

            current_score_text = points_to_tennis_score(points)
            if current_score_text == "Game":
                current_score_text = "Love"

            display_text = f"Games: {games} | Score: {current_score_text}"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Tennis Ball Tracker with Picamera2", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual.")
    finally:
        picamera.stop()
        cv2.destroyAllWindows()

def main():
    picamera = Picamera2()
    picamera.preview_configuration.main.size = (640, 480)
    picamera.preview_configuration.main.format = "RGB888"
    picamera.preview_configuration.align()
    picamera.configure("preview")
    picamera.start()

    process_stream(picamera, seconds_absence=1)

if __name__ == "__main__":
    main()
