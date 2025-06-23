import cv2
import numpy as np

def dibujar_linea_representativa(frame, lineas, color, height):
    """
    Calcula una línea promedio a partir de una lista de líneas y la dibuja en el frame.

    Args:
        frame (np.ndarray): Imagen original sobre la que se dibuja la línea.
        lineas (list): Lista de líneas, donde cada una es una tupla (x1, y1, x2, y2).
        color (tuple): Color en BGR de la línea a dibujar.
        height (int): Altura del frame, usada para definir los extremos verticales de la línea dibujada.
    
    """
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
    """
    Procesa un video para detectar y dibujar las líneas del carril por el que circula un auto.
    Si en algún frame no se detectan líneas en un lado, se reutilizan las del frame anterior.

    Args:
        ruta_video (str): Ruta al archivo de video a procesar.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {ruta_video}")
        return        

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el frame o fin del video")
            break
        
        # Obtenemos dimensiones
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Memoria de líneas anteriores
        lineas_izquierda_prev = []
        lineas_derecha_prev = []

        # Pasamos a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicamos Suavizado
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

        # Detección de bordes con Canny
        edges = cv2.Canny(gris_suave, 80, 150)

        # Aplicamos ROI
        edges_roi = cv2.bitwise_and(edges, mascara)

        # Mostrar Canny sin ROI y con ROI aplicado
        cv2.imshow("Canny sin ROI", edges)
        cv2.imshow("Canny con ROI aplicado", edges_roi)      

        # Detección de lineas con transformada de Hough
        lineas = cv2.HoughLinesP(
            edges_roi,
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
        
        # Si no hay nuevas líneas, usamos las del frame anterior
        if not lineas_izquierda:
            lineas_izquierda = lineas_izquierda_prev
        else:
            lineas_izquierda_prev = lineas_izquierda

        if not lineas_derecha:
            lineas_derecha = lineas_derecha_prev
        else:
            lineas_derecha_prev = lineas_derecha

        # Dibujar líneas representativas en el frame original
        dibujar_linea_representativa(frame, lineas_izquierda, (255, 0, 0), height)
        dibujar_linea_representativa(frame, lineas_derecha, (255, 0, 0), height)

        # Mostrar resultado
        cv2.imshow('Lineas representativas', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # pausa
            print("⏸ Video pausado. Presioná cualquier tecla para continuar.")
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

procesar_video("ruta_1.mp4")
# procesar_video("ruta_2.mp4")