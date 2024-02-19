import cv2

def capture_image():
    cap = cv2.VideoCapture(2)  # Abre la cámara
    ret, frame = cap.read()  # Captura un fotograma
    if ret:
        cv2.imwrite('webcam/image.png', frame)  # Guarda la imagen capturada
        cap.release()  # Libera la cámara
        cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV

# Llama a la función para capturar una imagen
capture_image()
