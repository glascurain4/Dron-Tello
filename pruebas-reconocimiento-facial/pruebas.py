import cv2
import numpy as np

def findFace(img):
    if img is None:
        print("Error: La imagen es None. Verifica la fuente de video.")
        return img, [[0, 0], 0]

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error al convertir a escala de grises: {e}")
        return img, [[0, 0], 0]
    
    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4)
    
    myFaceListC = []
    myFaceListArea = []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
        
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    img, info = findFace(img)
    print("Centro:", info[0], "Área:", info[1])  # Información del rostro detectado
    cv2.imshow("Output", img)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
