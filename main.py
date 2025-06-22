import cv2
import numpy as np

def punto_cerca_borde(mascara, x, y, margen=3):
    h, w = mascara.shape
    x_min = max(x - margen, 0)
    x_max = min(x + margen, w - 1)
    y_min = max(y - margen, 0)
    y_max = min(y + margen, h - 1)

    ventana = mascara[y_min:y_max+1, x_min:x_max+1]

    return np.any(ventana == 0) and np.any(ventana == 255)

def dibujar_linea_representativa(frame, lineas, color, height):
    if len(lineas) == 0:
        return
    xs = []
    ys = []
    for x1, y1, x2, y2 in lineas:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    m, b = np.polyfit(xs, ys, 1)

    y1_draw = height
    y2_draw = 330  # altura del vértice superior del trapecio ROI

    x1_draw = int((y1_draw - b) / m)
    x2_draw = int((y2_draw - b) / m)

    cv2.line(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 6)

def procesar_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {ruta_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame o fin del video")
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gris_suave = cv2.GaussianBlur(gris, (5, 5), 0)

        # Crear máscara de ROI (trapecio)
        mascara = np.zeros_like(gris_suave)
        puntos = np.array([[
            (115, height),
            (915, height),
            (560, 330),
            (420, 330)
        ]], dtype=np.int32)
        cv2.fillPoly(mascara, puntos, 255)

        roi = cv2.bitwise_and(gris_suave, mascara)
        edges = cv2.Canny(roi, threshold1=80, threshold2=150)

        lineas = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=50,
            maxLineGap=100
        )

        lineas_izquierda = []
        lineas_derecha = []

        if lineas is not None:
            for linea in lineas:
                x1, y1, x2, y2 = linea[0]

                if punto_cerca_borde(mascara, x1, y1) or punto_cerca_borde(mascara, x2, y2):
                    continue
                if x2 - x1 == 0:
                    continue

                pendiente = (y2 - y1) / (x2 - x1)
                if abs(pendiente) < 0.5:
                    continue

                x_centro = (x1 + x2) / 2
                if pendiente < 0 and x_centro < width / 2:
                    lineas_izquierda.append((x1, y1, x2, y2))
                elif pendiente > 0 and x_centro > width / 2:
                    lineas_derecha.append((x1, y1, x2, y2))

        # Dibujar líneas representativas en el frame original
        dibujar_linea_representativa(frame, lineas_izquierda, (255, 0, 0), height)
        dibujar_linea_representativa(frame, lineas_derecha, (255, 0, 0), height)

        # Mostrar resultado
        cv2.imshow('Lineas representativas', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#procesar_video("ruta_1.mp4")
procesar_video("ruta_2.mp4")