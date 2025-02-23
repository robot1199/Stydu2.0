import cv2
import numpy as np

# Функция для выбора точек с помощью мыши
selected_points = []
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Frame', frame)

# Открытие видеофайла
cap = cv2.VideoCapture('test_video.mp4')
ret, frame = cap.read()
if not ret:
    print("Ошибка: Не удалось открыть видеофайл.")
    exit()

# Остановка видео на первом кадре и выбор точек
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

print("Выберите точки (например, плечо, локоть, запястье). Нажмите 'q', чтобы продолжить.")
while True:
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(selected_points) >= 3:
        break

# Инициализация трекеров для выбранных точек
trackers = []
for (x, y) in selected_points:
    tracker = cv2.TrackerCSRT_create()
    bbox = (x - 15, y - 15, 30, 30)  # Область вокруг точки
    tracker.init(frame, bbox)
    trackers.append(tracker)

# Основной цикл для отслеживания точек
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обновление трекеров
    for tracker in trackers:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()