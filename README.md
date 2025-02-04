# 🚗🔍 IA-SMARTPARKING 🔍🚗
#### 💻 Detector de Placas y Reconocimiento Facial para Control de Estacionamiento con: 
> #### 🔹Tensorflow 🔹Python 🔹Tesseract-OCR.
> Este proyecto es una parte clave del código que me ayudó a obtener mi título en Ingeniería en Sistemas Computacionales. Su propósito es la detección de placas vehiculares y reconocimiento facial del conductor, permitiendo asociar ambos elementos para validar la identidad del usuario y garantizar la seguridad del estacionamiento.

>## 🚀 Características
>- 📌 - Detección de placas vehiculares mediante técnicas de procesamiento de imágenes y OCR.
>- 🗃️ - Reconocimiento facial del conductor para asociarlo con la placa del vehículo.
>- 🔑 - Validación de acceso para asegurar que solo conductores autorizados puedan ingresar.

#### 🌐 Visualización
---
> #### Funcionamiento Interno
> ![](/CosasREADME/Lectura-Placa.jpg)
> #### Video Entrada:
>[🎥 Ver video de entrada](https://github.com/ReyesPedrizcoAngelAudiel/DeteccionPlacas-Rostros/blob/master/CosasREADME/Entrada-Correcta.mp4)
> #### Video Salida Correcta:
>[🎥 Ver video de Salida Correcta](https://github.com/ReyesPedrizcoAngelAudiel/DeteccionPlacas-Rostros/blob/master/CosasREADME/Salida-Correcta.mp4)
> #### Video Salida Incorrecta:
>[🎥 Ver video de Salida Incorrecta](https://github.com/ReyesPedrizcoAngelAudiel/DeteccionPlacas-Rostros/blob/master/CosasREADME/Salida_Incorrecta.mp4)

#### ⚙️ Instalación
>- **Requisitos**
>   - **Python 3.12** o superior (Este proyecto fue desarrollado con Python 3.12.3)
>   - **Tesseract-OCR** (para la lectura de placas)
>- **Instalación de dependencias**
>   - Ejecuta el siguiente comando para instalar las librerías necesarias:

    pip install opencv-python numpy tensorflow-lite pytesseract scikit-image
>- **Instalación de Tesseract-OCR**
>- **📌 Windows**
>   - Descarga el instalador desde: https://github.com/UB-Mannheim/tesseract/wiki
>   - Durante la instalación, marca la opción "Add Tesseract to the system PATH".
>   - Una vez instalado, verifica la instalación ejecutando en la terminal:

	tesseract --version

>- **📌 Linux (Ubuntu/Debian)**

    sudo apt update
    sudo apt install tesseract-ocr -y

>- **📌 MacOS**
    
	brew install tesseract
---
###### 🌟 ¡Gracias por revisar este proyecto! 
###### Este sistema puede ser útil en estacionamientos inteligentes, control de accesos y seguridad vehicular. 🚀 🦊
