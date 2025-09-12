
import cv2


cap = cv2.VideoCapture(0)


net = cv2.dnn.readNetFromCaffe("opencv_face_detector.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

anchonet = 300
altonet = 300
media = [104, 117, 123]
umbral = 0.7

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    altoframe = frame.shape[0]
    anchoframe = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (anchonet, altonet), media, swapRB = False, crop = False)

    net.setInput(blob)
    detecciones = net.forward()

    for i in range(detecciones.shape[2]):
        conf_detect = detecciones[0,0,i,2]
        if conf_detect > umbral:
            xmin = int(detecciones[0, 0, i, 3] * anchoframe)
            ymin = int(detecciones[0, 0, i, 4] * altoframe)
            xmax = int(detecciones[0, 0, i, 5] * anchoframe)
            ymax = int(detecciones[0, 0, i, 6] * altoframe)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
            label = "Confianza de deteccion: %.4f" % conf_detect
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (xmin, ymin - label_size[1]), (xmin + label_size[0], ymin + base_line),
                          (0,0,0), cv2.FILLED)
            cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("DETECCION DE ROSTROS", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()