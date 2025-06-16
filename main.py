import cv2
import numpy as np


# Abrimos el video
cap = cv2.VideoCapture('ruta_2.mp4')                

# Obtenemos dimensiones del video para posteriormente usar en ROI
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Funcion para que no tome los bordes de ROI
def punto_cerca_borde(mascara, x, y, margen=3):
    """
    Verifica si el punto (x,y) está a menos de 'margen' pixeles de la frontera
    entre 0 y 255 en la máscara.

    Parámetros:
    - mascara: array numpy 2D con valores 0 o 255
    - x,y: coordenadas del punto
    - margen: radio de búsqueda para vecinos (por defecto 3 píxeles)

    Retorna:
    - True si está cerca del borde, False si está dentro del área sin borde
    """
    h, w = mascara.shape
    x_min = max(x - margen, 0)
    x_max = min(x + margen, w - 1)
    y_min = max(y - margen, 0)
    y_max = min(y + margen, h - 1)

    ventana = mascara[y_min:y_max+1, x_min:x_max+1]

    # Si dentro de la ventana hay tanto pixeles 0 como 255, está en la frontera
    if np.any(ventana == 0) and np.any(ventana == 255):
        return True
    return False


# Procesamos el video
while cap.isOpened():                         # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                   # Obtengo el frame
    if ret:                                   # ret indica si la lectura fue exitosa (True) o no (False)
        # Mostramos el frame original
        cv2.imshow('Frame',frame)
        
        # Convertimos a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gris', gris)
        
        gris_suave = cv2.GaussianBlur(gris, (5, 5), 0) # suavizamos para que no se detecten bordes falsos
        cv2.imshow('Gris Suave', gris_suave)
        
        # Creamos una máscara para la región de interés (ROI)
        mascara = np.zeros_like(gris_suave)
        puntos = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)]], dtype=np.int32)
        cv2.fillPoly(mascara, puntos, 255)

        # Aplicamos la máscara
        roi = cv2.bitwise_and(gris_suave, mascara)
        
        cv2.imshow('ROI aplicada', roi) # esto para mostrar como queda ROI


        # Usamos Canny para detectar bordes 
        edges = cv2.Canny(roi, threshold1=80, threshold2=150)
        
        cv2.imshow('Bordes Canny', edges)


        # Usamos Hough para detectar líneas
        lineas = cv2.HoughLinesP(
        edges,
        rho=1,                     # precisión en pixeles
        theta=np.pi/180,           # precisión angular (1 grado)
        threshold=20,              # mínimo votos para considerar una línea
        minLineLength=40,          # longitud mínima de línea
        maxLineGap=100             # máxima separación entre segmentos para unirlos
        )
        
        # Dibujamos las líneas detectadas 
        if lineas is not None:
            for linea in lineas:
                x1, y1, x2, y2 = linea[0]
                
                # Si alguno de los puntos está cerca del borde, descartamos la línea
                if punto_cerca_borde(mascara, x1, y1) or punto_cerca_borde(mascara, x2, y2):
                    continue
                
                # Si pasa el filtro, dibujamos la línea en el frame original
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # líneas azules, grosor 5

        cv2.imshow('Lineas detectadas', frame)
        
        # Esperamos tecla para salir
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print("Error al leer el frame")                                 # Imprimo un mensaje de error si no se pudo leer el frame
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.

# Liberamos recursos
cap.release()
cv2.destroyAllWindows()