
import cv2
import numpy as np
import mediapipe as mp
import math
import time

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9,
    model_complexity=2
)

# Данные спортсмена
body_mass = 90  # Масса тела в кг
length_of_leg = 0.5  # Длина голени в метрах

# Параметры для детектора весел
LOWER_COLOR = np.array([20, 100, 100])  # Нижний предел цвета весел в HSV
UPPER_COLOR = np.array([40, 255, 255])  # Верхний предел цвета весел в HSV

# Открытие видеофайла
cap = cv2.VideoCapture('test_video.mp4')  # Для видеофайла
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл.")
    exit()

frames = []  # Список для хранения кадров

# Определение соединений между ключевыми точками
connections = [
    (11, 13),  # Плечи к локтям
    (13, 15),  # Локти к запястьям
    (11, 23),  # Плечи к бедрам
    (23, 25),  # Бедра к коленям
    (25, 27)  # Колени к стопам
]

# Параметры углов таз-колено-стопа
time_interval = 0.0333  # Время между кадрами в секундах
previous_angle = 30  # Начальный угол в градусах
angular_velocities = []
angular_accelerations = []
knee_load = 0  # Нагрузка на колено

# Функции для расчетов
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def calculate_angle2(a, b, c):
    return calculate_angle(a, b, c)

def calculate_spine_deviation(spine_points):
    return calculate_angle(spine_points[0], spine_points[2], spine_points[4])

def calculate_center_of_mass_shift(body_parts, boat_center):
    center_of_mass = np.mean(body_parts, axis=0)
    return np.linalg.norm(center_of_mass - boat_center)

def calculate_torso_center(landmarks):
    left_hip = landmarks[23].x, landmarks[23].y
    right_hip = landmarks[24].x, landmarks[24].y
    left_shoulder = landmarks[11].x, landmarks[11].y
    right_shoulder = landmarks[12].x, landmarks[12].y
    return np.mean([left_hip, right_hip, left_shoulder, right_shoulder], axis=0)

# Основной цикл обработки видео
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret or len(frames) > 200:
        break

    frame = cv2.resize(frame, (800, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Извлечение координат ключевых точек
        left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        left_elbow = [landmarks[13].x * w, landmarks[13].y * h]
        left_wrist = [landmarks[15].x * w, landmarks[15].y * h]
        left_hip = [landmarks[23].x * w, landmarks[23].y * h]
        left_knee = [landmarks[25].x * w, landmarks[25].y * h]
        left_ankle = [landmarks[27].x * w, landmarks[27].y * h]

        # Вычисление углов
        hip_knee_ankle_angle = calculate_angle(left_hip, left_knee, left_ankle)
        knee_hip_shoulder_angle = calculate_angle(left_knee, left_hip, left_shoulder)
        shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        knee_ankle_angle = calculate_angle2(left_knee, left_ankle, [left_ankle[0] + 100, left_ankle[1]])
        shoulder_hip_angle = calculate_angle2(left_shoulder, left_hip, [left_hip[0] - 100, left_hip[1]])

        # Расчет нагрузки на колено
        current_angle = hip_knee_ankle_angle
        current_angle_radians = np.radians(current_angle)
        previous_angle_radians = np.radians(previous_angle)
        delta_theta = current_angle_radians - previous_angle_radians
        omega = delta_theta / time_interval
        angular_velocities.append(omega)

        if len(angular_velocities) > 1:
            delta_omega = angular_velocities[-1] - angular_velocities[-2]
            alpha = delta_omega / time_interval
            angular_accelerations.append(alpha)
            linear_acceleration = length_of_leg * alpha
            g = 9.81
            knee_load = (body_mass * (g + linear_acceleration * np.sin(current_angle_radians))) / 9.81

        previous_angle = current_angle

        # Отображение результатов
        panel = np.zeros((400, 800, 3), dtype=np.uint8)
        front_scale = 0.7
        cv2.putText(panel, f'Shoulder-Elbow-Wrist: {int(shoulder_elbow_wrist_angle)} degrees',
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Shoulder-Hip-Knee: {int(knee_hip_shoulder_angle)} degrees',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Hip-Knee-Ankle: {int(hip_knee_ankle_angle)} degrees',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Shoulder-Hip Angle: {int(shoulder_hip_angle)} degrees',
                    (10, 250), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, f'Knee-Ankle Angle: {int(knee_ankle_angle)} degrees',
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(panel, f"Knee Load: {knee_load:.1f} KG",
                    (10, 300), cv2.FONT_HERSHEY_SIMPLEX, front_scale, (0, 255, 0), 2, cv2.LINE_AA)

        # Отображение точек и линий
        for connection in connections:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
            end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        frame = np.vstack((frame, panel))
        frames.append(frame)

cap.release()

# Воспроизведение видео
current_frame = 0
playback_speed = 33
is_playing = True

window_name = 'Video'

while True:
    if is_playing:
        cv2.imshow(window_name, frames[current_frame])
        current_frame = (current_frame + 1) % len(frames)

    key = cv2.waitKey(playback_speed)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        is_playing = not is_playing
    elif key & 0xFF == ord('b'):
        current_frame = max(0, current_frame - 1)
    elif key & 0xFF == ord('f'):
        current_frame = min(len(frames) - 1, current_frame + 1)
    elif key & 0xFF == ord('+'):
        playback_speed = max(1, playback_speed -10)
    elif key & 0xFF == ord('-'):
        playback_speed += 10

        # Обновление окна, если видео не воспроизводится
    if not is_playing:
        cv2.imshow(window_name, frames[current_frame])

cv2.destroyAllWindows()











