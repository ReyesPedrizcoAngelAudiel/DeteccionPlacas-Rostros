import cv2
import numpy as np
import pytesseract
import re
from tensorflow.lite.python.interpreter import Interpreter
from skimage.segmentation import clear_border
import threading

# Configura la ruta al ejecutable de Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Expresión regular para encontrar placas de vehículos
placa_regex = r'^[A-Z]{3}-[0-9]{3}-[A-Z]$'

# Variable global para llevar el conteo de placas detectadas | MAXIMO MANDAR 5
placa_detectada_count = 0
# Cargar el clasificador en cascada de OpenCV para la detección de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#1. Función para realizar la detección de placas utilizando OCR en paralelo
def detect_plate_ocr(placa_area):
    global placa_detectada_count  # Usar la variable global

    # Filtro #1 | Convertir a escala de grises
    escala_grises = cv2.cvtColor(placa_area, cv2.COLOR_BGR2GRAY)
    # Filtro #2 | Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(escala_grises, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 133, 2)
    # Filtro #3 | Eliminar los bordes usando skimage
    sin_bordes = clear_border(thresh)
    # Filtro adicional | Morfología para eliminar ruido
    kernel = np.ones((2, 2), np.uint8)
    sin_ruido = cv2.morphologyEx(sin_bordes, cv2.MORPH_OPEN, kernel)
    # Filtro #4 | Invertir colores
    invertida = cv2.bitwise_not(sin_ruido)
    # Filtro adicional | Escalar la imagen para mejorar precisión del OCR
    invertida = cv2.resize(invertida, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Realizar OCR
    texto_placa = pytesseract.image_to_string(invertida, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip().replace('\n', '').replace(' ', '')

    # Verificar si el texto coincide con la expresión regular
    if re.match(placa_regex, texto_placa):
        # Si no se ha alcanzado el límite de prints
        if placa_detectada_count < 1:
            # Mostrar la imagen procesada en una ventana llamada "Placa"
            print("Placa detectada:", texto_placa)
            placa_detectada_count += 1  # Incrementar el contador
        return texto_placa
    return invertida

#2. Función para realizar detección con el modelo TFLite y mostrar los resultados en un video
def tflite_detect_video(modelpath, videopath, lblpath, min_conf=0.99, original_fps=1800.0, desired_fps=60.0):

    # Cargar el mapa de etiquetas
    with open(lblpath, 'r') as f: labels = [line.strip() for line in f.readlines()]

    # Cargar el modelo TFLite
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Obtener detalles del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Calcular cuántos frames saltar para alcanzar la tasa deseada
    frame_skip = int(original_fps / desired_fps)
    delay = int(1000 / desired_fps)

    # Abrir el video
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    frame_count = 0

    while True:
        # Leer el siguiente cuadro del video
        ret, frame = cap.read()

        if not ret: break  # Terminar si no hay más cuadros

        # Saltar frames para ajustar la tasa de FPS deseada
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Convertir la imagen a RGB y redimensionarla
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalizar los valores de los píxeles si se usa un modelo flotante
        float_input = (input_details[0]['dtype'] == np.float32)
        if float_input:
            input_mean = 127.5
            input_std = 127.5
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Realizar la detección ejecutando el modelo con la imagen como entrada
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Recuperar los resultados de la detección
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Dibujar las cajas de detección sobre la imagen
        for i in range(len(scores)):
            if scores[i] > min_conf:
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                # Obtener el nombre del objeto detectado
                object_name = labels[int(classes[i])]

                # Dibujar el rectángulo azul alrededor del objeto
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                # Dibujar la etiqueta con el nombre del objeto y la confianza
                label = '%s: %.2f%%' % (object_name, scores[i] * 100)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Verificar si se detectó un "ROSTRO"
                if object_name == "ROSTRO":
                    # Extraer el área del rostro
                    rostro_area = frame[ymin:ymax, xmin:xmax]
                    # Redimensionar el área del rostro a (150, 150)
                    rostro_resized = cv2.resize(rostro_area, (150, 150))
                    rostro_gris = cv2.cvtColor(rostro_resized, cv2.COLOR_BGR2GRAY)

                    faces = faceClassif.detectMultiScale(rostro_gris, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                    # Mostrar el rostro redimensionado en una ventana llamada "Rostro"
                    cv2.imshow("Rostro", rostro_resized)

                # Extraer la región de la placa para OCR
                elif object_name == "PLACA":
                    placa_area = frame[ymin:ymax, xmin:xmax]
                    thread = threading.Thread(target=detect_plate_ocr, args=(placa_area,))
                    thread.start()

        # Mostrar el cuadro con detección
        cv2.imshow("Video", frame)

        # Incrementar el contador de frames
        frame_count += 1

        # Salir del video si se presiona la tecla 'q'
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Cerrar el video y las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()

modelpath = 'detect.tflite'  # Ruta del modelo TFLite
lblpath = 'labelmap.txt'  # Ruta del archivo con las etiquetas del modelo

""" ----------------------------------- CARRITO DE ANGEL -------------------------------------------------"""
videopath = 'Videos/Entradas-Correctas/Entrada_Ang_C.mp4'        # Ruta del video de Entrada

""" ----------------------------------- CARRITO DE ELIZABETH -------------------------------------------------"""
#videopath = 'Videos/Entradas-Correctas/Entrada_Eli_C.mp4'        # Ruta del video de Entrada

""" ----------------------------------- CARRITO DE ARMIDA -------------------------------------------------"""
#videopath = 'Videos/Entradas-Correctas/Entrada_Armida_C.mp4'     # Ruta del video de Entrada

""" ----------------------------------- CARRITO DE FANNY -------------------------------------------------"""
#videopath = 'Videos/Entradas-Correctas/Entrada_Fanny_C.mp4'      # Ruta del video de Entrada

""" ----------------------------------- CARRITO DE ITZEL -------------------------------------------------"""
#videopath = 'Videos/Entradas-Correctas/Entrada_Itzel_C.mp4'      # Ruta del video de Entrada

""" ----------------------------------- CARRITO DE COCO -------------------------------------------------"""
#videopath = 'Videos/Entradas-Correctas/Entrada_Coco_C.mp4'       # Ruta del video de Entrada

# Ejecutar la detección para el video
tflite_detect_video(modelpath, videopath, lblpath)