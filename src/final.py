import cv2
import numpy as np
from picamera2 import Picamera2
import time
from tracker import *
from security_code import *


def process_stream(picamera, video_writer, kalman, seconds_absence=1):
    """Procesa el stream en tiempo real usando Picamera2 para realizar el tracking."""
    fps = 30  # Estimamos 30 FPS para la Picamera2
    absence_frames_threshold = int(seconds_absence * fps)

    lower_green = np.array([29, 86, 6], dtype="uint8")
    upper_green = np.array([64, 255, 255], dtype="uint8")

    points = 0
    games = 0
    absent_frames = 0
    already_counted = False

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

            # Escribir el fotograma en el video_writer para grabar toda la secuencia
            resized_frame = cv2.resize(frame, (640, 480))
            video_writer.write(resized_frame)

            cv2.imshow("Tennis Ball Tracker with Picamera2", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual por teclado.")
    finally:
        picamera.stop()
        video_writer.release()  # Detén la grabación aquí
        cv2.destroyAllWindows()


def main():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    
    color_ranges = {
        "morado": ((130, 50, 50), (160, 255, 255)),
        "amarillo": ((20, 100, 100), (30, 255, 255)),
        "azul": ((100, 150, 0), (140, 255, 255)),
        "verde": ((40, 100, 100), (80, 255, 255)),
    }
    
    unlock_sequence = ["Circulo morado", "Triangulo amarillo", "Circulo azul", "Rectangulo verde"]
    detected_sequence = []
    last_detected_shape = None
    message = ""
    message_start_time = 0
    message_color = (0, 255, 255)  # Amarillo
    last_detection_time = 0  # Inicializar la variable de tiempo
    min_area = 10000

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("/home/pi/Desktop/ProyectoFinal_CV/data/videos/outputVideos/video_final_incorrect.mp4", fourcc, 30, (640, 480))

    prev_time = time.time()  # Para calcular FPS
    kalman = init_kalman_filter()
    
    try:
        # Comienza la grabación antes de entrar al bucle principal
        while True:
            frame = picam.capture_array()
            detected_colored_shapes = detect_colored_shapes(frame, color_ranges, min_area)

            current_time = time.time()

            # Calcular FPS
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            for colored_shape in detected_colored_shapes:
                if len(detected_sequence) < len(unlock_sequence) and colored_shape != last_detected_shape:
                    # Verificar si han pasado al menos 3 segundos desde la última detección
                    if current_time - last_detection_time >= 3:
                        last_detected_shape = colored_shape
                        detected_sequence.append(colored_shape)
                        message = f"Detectado: {colored_shape}"
                        print(message)  # Mensaje por terminal
                        print("Detected sequence: ", detected_sequence)
                        message_start_time = current_time
                        message_color = (0, 255, 255)  # Amarillo
                        last_detection_time = current_time  # Actualizar el tiempo de la última detección

                # Evaluar la secuencia al alcanzar 4 figuras
                if len(detected_sequence) == len(unlock_sequence):
                    if detected_sequence == unlock_sequence:
                        message = "Desbloqueo exitoso"
                        print(message)  # Mensaje por terminal
                        process_stream(picam, video_writer, kalman)  # Aquí se activa el tracker en la misma ventana
                        return
                    else:
                        message = "Desbloqueo erroneo. Reiniciando..."
                        print(message)  # Mensaje por terminal
                        message_color = (0, 0, 255)  # Rojo
                    message_start_time = current_time
                    detected_sequence = []

            # Mostrar la secuencia actual
            sequence_text = f"Secuencia: {' -> '.join(detected_sequence)}"
            cv2.putText(frame, sequence_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Mostrar mensaje por 2 segundos
            if message and current_time - message_start_time < 2:
                cv2.putText(frame, message, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, message_color, 3)
            elif message and current_time - message_start_time >= 2:
                message = ""

            # Mostrar FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Escribir el fotograma en el video_writer para grabar toda la secuencia
            resized_frame = cv2.resize(frame, (640, 480))
            video_writer.write(resized_frame)
            cv2.imshow("Deteccion de Formas y Colores / Tracker", resized_frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual por teclado.")
    finally:
        picam.stop()
        video_writer.release()  # Detén la grabación aquí
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



    
