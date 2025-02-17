import cv2
import numpy as np

# Захват видео с веб-камеры
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('train.mp4')

# Параметры для алгоритма Lucas-Kanade
feature_params = dict(
    maxCorners=100,  # Максимальное количество углов для отслеживания
    qualityLevel=0.3,  # Минимальное качество углов
    minDistance=7,  # Минимальное расстояние между углами
    blockSize=7  # Размер блока для вычисления углов
)

# Параметры для оптического потока
lk_params = dict(
    winSize=(15, 15),  # Размер окна для поиска
    maxLevel=2,  # Уровни пирамиды
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Критерии остановки
)

# Чтение первого кадра
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Поиск углов для отслеживания
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Создание маски для визуализации
mask = np.zeros_like(old_frame)

while True:
    # Чтение нового кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование в оттенки серого
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Вычисление оптического потока
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Выбор хороших точек (те, которые успешно отслежены)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # Рисование треков
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Новые координаты
        c, d = old.ravel()  # Старые координаты
        # Преобразование координат в целые числа
        a, b, c, d = int(a), int(b), int(c), int(d)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)  # Линия между точками
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)  # Круг на новой точке

    # Наложение маски на кадр
    img = cv2.add(frame, mask)

    # Отображение результата
    cv2.imshow('Optical Flow', img)

    # Обновление предыдущего кадра и точек
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
