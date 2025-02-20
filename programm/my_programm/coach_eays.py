import cv2
import numpy as np
import mediapipe as mp


# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    model_complexity=2  # Повышенная точность для ног
)


# Параметры для детектора весел
LOWER_COLOR = np.array([20, 100, 100]) # Подстроить под цвет весел
UPPER_COLOR = np.array([40, 255, 255])




# Открытие видеофайла
cap = cv2.VideoCapture('test_video.mp4') # Для видеофайла
# cap = cv2.VideoCapture(0)  # Для веб-камеры
frames = []

# Определение соединений между ключевыми точками
connections = [
    (11, 13),  # Плечи к локтям
    (13, 15),  # Локти к запястьям
    (11, 23),  # Плечи к бедрам
    (23, 25),  # Бедра к коленям
    (25, 27)  # Колени к стопам
]

# Сохранение всех кадров в список
while True:
    ret, frame = cap.read()
    if not ret or len(frames) > 200:
        break

    # Преобразование цвета для MediaPipe
    frame = cv2.resize(frame, (1280, 720))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Отображение результатов на кадре
    if results.pose_landmarks:
        # Индексы точек, которые мы хотим оставить
        keep_indices = [11, 13, 15, 23, 25, 27]  # Голова, плечи, бедра
        filtered_landmarks = [results.pose_landmarks.landmark[i] for i in keep_indices]

        # Отображение отфильтрованных точек
        for landmark in filtered_landmarks:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Зеленые точки

        # Соединение точек
        for connection in connections:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Красные линии

    frames.append(frame)

cap.release()

# Переменные для управления воспроизведения
current_frame = 0
playback_speed = 100 # задержка в милисекундах(100 мс = нормальная скорость)
is_playing = True
print(f'playback_speed = {playback_speed}')


# установка размера окна
window_name = 'Video'


while True:
    if is_playing:
        # Отображение текущего кадра
        cv2.imshow(window_name, frames[current_frame])
        current_frame = (current_frame + 1) % len(frames)  # Циклический переход к началу

    key = cv2.waitKey(playback_speed)  # Задержка для воспроизведения

    if key & 0xFF == ord('q'):  # Выход
        break
    elif key & 0xFF == ord('s'):  # Остановка
        is_playing = not is_playing  # Переключение состояния воспроизведения
        print('stop/go')
    elif key & 0xFF == ord('b'):  # Возврат на один кадр
        current_frame = max(0, current_frame - 1)
        print('recovery')
    elif key & 0xFF == ord('f'):  # Вперед на один кадр
        current_frame = min(len(frames) - 1, current_frame + 1)
        print('plus one steep')
    elif key & 0xFF == ord('+'):  # Ускорение
        playback_speed = max(1, playback_speed - 10)  # Уменьшение задержки
        print(f'acceleration playback_speed = {playback_speed}')
    elif key & 0xFF == ord('-'):  # Замедление
        playback_speed += 10  # Увеличение задержки
        print(f'deceleration playback_speed = {playback_speed}')

        # Обновление окна, если видео не воспроизводится
    if not is_playing:
        cv2.imshow(window_name, frames[current_frame])




# Освобождение ресурсов
cv2.destroyAllWindows()