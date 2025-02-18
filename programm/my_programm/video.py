import cv2

# Открытие видеофайла
video = cv2.VideoCapture('test2.mp4')

while video.isOpened():
    ret, frame = video.read()  # Чтение кадра
    if not ret:
        break

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video.release()
cv2.destroyAllWindows()