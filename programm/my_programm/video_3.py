import cv2

# Открытие видеофайла
video = cv2.VideoCapture('mingechavir.mp4')

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Применение размытия
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Отображение кадра
    cv2.imshow('Blurred Video', blurred_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()