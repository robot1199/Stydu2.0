import cv2
import numpy as np
import mediapipe as mp
from function import *

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.9,  # Уровень уверенности для обнаружения
    min_tracking_confidence=0.9,  # Уровень уверенности для отслеживания
    model_complexity=2  # Повышенная точность для ног
)

# Параметры для детектора весел
LOWER_COLOR = np.array([20, 100, 100])  # Нижний предел цвета весел в HSV
UPPER_COLOR = np.array([40, 255, 255])  # Верхний предел цвета весел в HSV

# Открытие видеофайла
cap = cv2.VideoCapture('test_video.mp4')  # Для видеофайла
# cap = cv2.VideoCapture(0)  # Для веб-камеры
frames = []  # Список для хранения кадров

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
    ret, frame = cap.read()  # Чтение кадра из видео
    if not ret or len(frames) > 200:  # Проверка на конец видео или превышение количества кадров
        break

    # Преобразование цвета для MediaPipe
    frame = cv2.resize(frame, (1280, 720))  # Изменение размера кадра
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Преобразование в RGB
    results = pose.process(frame_rgb)  # Обработка кадра с помощью MediaPipe

    # Отображение результатов на кадре
    if results.pose_landmarks:
        # Индексы точек, которые мы хотим оставить
        keep_indices = [11, 13, 15, 23, 25, 27]  # Индексы ключевых точек
        filtered_landmarks = [results.pose_landmarks.landmark[i] for i in keep_indices]

        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape  # Размеры кадра

        # Извлечение координат ключевых точек
        left_hip = [landmarks[23].x * w, landmarks[23].y * h]  # Левое бедро
        left_knee = [landmarks[25].x * w, landmarks[25].y * h]  # Левое колено
        left_ankle = [landmarks[27].x * w, landmarks[27].y * h]  # Левая стопа

        left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]  # Левое плечо
        left_elbow = [landmarks[13].x * w, landmarks[13].y * h]  # Левый локоть
        left_wrist = [landmarks[15].x * w, landmarks[15].y * h]  # Левая кисть

        # Определение точки на "полу" (горизонтальная линия)
        right_line_end = [left_ankle[0] + 100, left_ankle[1]]  # Точка, чтобы нарисовать линию вправо от стопы
        left_line_end = [left_hip[0] - 100, left_hip[1]]  # Точка, чтобы нарисовать линию влево от бедра


        # Вычисление углов
        hip_knee_ankle_angle = calculate_angle(left_hip, left_knee, left_ankle)  # Бедро-колено-стопа
        knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)  # Колено-бедро-плечо
        shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)  # Плечо-локоть-кисть
        knee_ankle_angle = calculate_angle2(left_knee, left_ankle, right_line_end)  # Колено-Стопа
        shoulder_hip_angle = calculate_angle2(left_shoulder, left_hip, left_line_end)  # Плечо-Бедро


        # Создание текстовой панели для отображения углов
        panel = np.zeros((300, 1280, 3), dtype=np.uint8)  # Черная панель внизу видео

        front_scale = 1
        # Вывод углов на текстовой панели
        cv2.putText(panel, f'Hip-Knee-Ankle: {int(hip_knee_ankle_angle)} degrees',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Knee-Hip-Shoulder Angle: {int(knee_hip_shoulder_angle)} degrees',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Shoulder-Elbow-Wrist: {int(shoulder_elbow_wrist_angle)} degrees',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Knee-Ankle Angle: {int(knee_ankle_angle)} degrees',
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Shoulder-Hip Angle: {int(shoulder_hip_angle)} degrees',
                    (10, 250), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (255, 255, 255), 2, cv2.LINE_AA)


        # Отображение отфильтрованных точек
        for landmark in filtered_landmarks:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)  # Преобразование нормализованных координат
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Отображение зеленых точек

        # Соединение точек
        for connection in connections:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Отображение красных линий

            # Рисуем линию, параллельную полу, уходящую вправо от стопы
            cv2.line(frame, (int(left_ankle[0]), int(left_ankle[1])), (int(left_ankle[0] + 100), int(left_ankle[1])),
                     (0, 255, 0), 2)  # Линия от стопы, параллельная полу

            # Рисуем линию, уходящую влево от бедра
            cv2.line(frame, (int(left_hip[0]), int(left_hip[1])), (int(left_hip[0] - 100), int(left_hip[1])),
                     (0, 255, 0), 2)  # Линия от бедра, уходящая влево

            # Рисуем линию от колена до стопы
            cv2.line(frame, (int(left_knee[0]), int(left_knee[1])), (int(left_ankle[0]), int(left_ankle[1])),
                     (255, 0, 0), 2)  # Линия от колена до стопы

            # Рисуем линию от плеча до бедра
            cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_hip[0]), int(left_hip[1])),
                     (255, 0, 0), 2)  # Линия от плеча до бедра

        torso_center = calculate_torso_center(results.pose_landmarks.landmark)

        # Объединение видео и текстовой панели
        frame = np.vstack((frame, panel))  # Добавляем панель внизу кадра

    frames.append(frame)  # Сохранение кадра в список

cap.release()  # Освобождение ресурсов

# Переменные для управления воспроизведения
current_frame = 0
playback_speed = 100  # Задержка в миллисекундах (100 мс = нормальная скорость)
is_playing = True
print(f'playback_speed = {playback_speed}')

# Установка размера окна
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
        print('plus one step')
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
