import cv2

# Открытие видеофайла
video = cv2.VideoCapture('mingechavir.mp4')

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Преобразование в оттенки серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Отображение кадра
    cv2.imshow('Gray Video', gray_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()