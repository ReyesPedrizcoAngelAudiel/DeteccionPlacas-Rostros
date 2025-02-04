import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
# Lectura de placas
import pytesseract
from skimage.segmentation import clear_border
import re

# Configura la ruta al ejecutable de Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Expresión regular para encontrar placas de vehículos
placa_regex = r'^[A-Z]{3}-[0-9]{3}-[A-Z]$'

# Función para mostrar el área de la detección de la placa y realizar OCR
def detect_plate_ocr(image, xmin, ymin, xmax, ymax):
    # Recortar el área de la imagen correspondiente a la detección
    plate_region = image[ymin:ymax, xmin:xmax]

    # Filtro #1 | Convertir a escala de grises
    escala_grises = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

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
    invertida = cv2.resize(invertida, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Realizar OCR
    texto_placa = pytesseract.image_to_string(invertida, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-').strip().replace('\n', '').replace(' ', '')

    # Mostrar el texto detectado en la terminal
    #print("Texto extraído por OCR:", texto_placa)

    # Verificar si el texto coincide con la expresión regular
    if re.match(placa_regex, texto_placa):
        print("Placa válida detectada:", texto_placa)
    else:
        print("Texto detectado no es una placa válida")

    # Mostrar la imagen procesada en una ventana llamada "Placa"
    cv2.imshow("Placa", invertida)

# Función para realizar detección con el modelo TFLite y mostrar los resultados
def tflite_detect_single_image(modelpath, imgpath, lblpath, min_conf=0.8):

    # Cargar el mapa de etiquetas
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Cargar el modelo TFLite
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Obtener detalles del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Cargar la imagen y redimensionarla al tamaño esperado por el modelo
    image = cv2.imread(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalizar los valores de los píxeles si se usa un modelo flotante (no cuantizado)
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

            # Dibujar el rectángulo azul alrededor del objeto
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

            # Dibujar la etiqueta con el nombre del objeto y la confianza
            object_name = labels[int(classes[i])]
            label = '%s: %.2f%%' % (object_name, scores[i] * 100)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Llamar a la función detect_plate_ocr para mostrar el área de la detección
            if "placa" in object_name.lower():  # Filtrar solo objetos que contengan "placa" en su nombre
                detect_plate_ocr(image, xmin, ymin, xmax, ymax)

    # Mostrar la imagen con OpenCV
    cv2.imshow("Deteccion", image)

    # Esperar por una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Parámetros del modelo y las rutas
modelpath = 'detect.tflite'
imgpath = 'Imagenes/Entrada/Imagen-10.jpg'
lblpath = 'labelmap.txt'

# Ejecutar la detección para una imagen específica
tflite_detect_single_image(modelpath, imgpath, lblpath, min_conf=0.8)