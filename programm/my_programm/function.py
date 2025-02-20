import numpy as np




# Функция для вычисления угла между тремя точками
def calculate_angle2(a, b, c):
    a = np.array(a)  # Первая точка
    b = np.array(b)  # Вершина угла
    c = np.array(c)  # Вторая точка

    # Вычисляем векторы
    ab = b - a
    bc = c - b

    # Вычисляем угол в радианах
    angle = np.arctan2(bc[1], bc[0]) - np.arctan2(ab[1], ab[0])
    angle = np.abs(angle * 180.0 / np.pi)  # Преобразуем в градусы

    # Убедимся, что угол не превышает 180 градусов
    if angle > 180.0:
        angle = 360 - angle

    return 180 - angle

# Функция для вычисления угла между тремя точками
def calculate_angle(a, b, c):
    a = np.array(a)  # Первая точка
    b = np.array(b)  # Вершина угла
    c = np.array(c)  # Вторая точка

    # Вычисляем угол в радианах
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)  # Преобразуем в градусы

    # Убедимся, что угол не превышает 180 градусов
    if angle > 180.0:
        angle = 360 - angle

    return angle


# Функция для вычисления координат виртуальной контрольной точки в центре туловища
def calculate_torso_center(landmarks):
    left_shoulder = landmarks[11]  # Левое плечо
    left_hip = landmarks[23]        # Левое бедро

    # Вычисляем средние координаты
    center_x = (left_shoulder.x + left_hip.x) / 4
    center_y = (left_shoulder.y + left_hip.y) / 4

    return int(center_x), int(center_y)