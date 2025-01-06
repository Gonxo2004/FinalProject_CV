import cv2
import numpy as np
from picamera2 import Picamera2
import time

# FUNCIONES AUXILIARES
def detect_shape(contour):
    """Determina la forma de un contorno (triángulo, cuadrado, círculo, rectángulo)."""
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 3:
        return "Triangulo"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.9 <= aspect_ratio <= 1.1:
            return "Cuadrado"
        else:
            return "Rectangulo"
    elif len(approx) >= 8:
        return "Circulo"
    return None  # Retorna None si no se identifica la forma

def segment_color(frame, lower_bound, upper_bound):
    """Segmenta un color específico en la imagen."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, segmented

def process_frame_for_shapes(frame, min_area=1000):
    """Procesa un fotograma para detectar formas geométricas y colores específicos."""
    detected_shapes = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            shape = detect_shape(contour)
            if shape:  # Evita figuras None
                detected_shapes.append((shape, area))
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, f"{shape} ({int(area)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, detected_shapes

def detect_colored_shapes(frame, color_ranges, min_area=1000):
    detected_colored_shapes = []

    for color_name, (lower, upper) in color_ranges.items():
        mask, segmented = segment_color(frame, lower, upper)
        _, detected_shapes = process_frame_for_shapes(segmented, min_area)

        for shape, area in detected_shapes:
            detected_colored_shapes.append(f"{shape} {color_name}")

    return detected_colored_shapes

# PROCESAMIENTO EN TIEMPO REAL
if __name__ == "__main__":
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    min_area = 10000

    color_ranges = {
        "morado": ((130, 50, 50), (160, 255, 255)),  # Rango ajustado para el color morado
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

    try:
        while True:
            frame = picam.capture_array()
            detected_colored_shapes = detect_colored_shapes(frame, color_ranges, min_area)

            current_time = time.time()

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
                        message_color = (0, 255, 0)  # Verde
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

            cv2.imshow("Deteccion de Formas y Colores", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción manual por teclado.")
    finally:
        picam.stop()
        cv2.destroyAllWindows()






