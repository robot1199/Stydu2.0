import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
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


# Функция для вычисления положения звена на основе угла и длины
def calculate_link_position(base_position, angle, length):
    x = base_position[0] + length * np.cos(np.radians(angle))
    y = base_position[1] + length * np.sin(np.radians(angle))
    return [x, y]


# Функция для вычисления отклонения позвоночника от вертикали
def calculate_spine_deviation(spine_points):
    # spine_points - список координат точек вдоль позвоночника
    # Например, [top_of_head, neck, upper_back, lower_back]

    # Вычисляем среднюю позицию позвоночника
    spine_midpoint = np.mean(spine_points, axis=0)

    # Определяем вертикальную линию (например, по координате x)
    vertical_line = np.array([spine_midpoint[0], 0])  # y = 0 для вертикали

    # Вычисляем угол отклонения
    angle_deviation = np.arctan2(spine_midpoint[1], spine_midpoint[0]) * (180 / np.pi)

    return angle_deviation


# Функция для вычисления смещения центра масс относительно центра лодки
def calculate_center_of_mass_shift(body_parts, boat_center):
    # body_parts - список координат ключевых точек тела
    # boat_center - координаты центра лодки

    # Вычисляем центр масс спортсмена
    center_of_mass = np.mean(body_parts, axis=0)

    # Вычисляем смещение центра масс относительно центра лодки
    shift_vector = center_of_mass - boat_center

    return shift_vector

# функция для создания графиков
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plot_knee_load(history):
    fig, ax = plt.subplots(figsize=(8, 2))  # создаем фигуру и оси
    ax.plot(history, color='green')  # рисуем график
    ax.set_title('Knee Load Over Time')  # Заголовок
    ax.set_xlabel('Frame')
    ax.set_ylabel('KG')
    ax.grid(True)  # включаем сетку

    # Преобразуем график в изображение
    canvas = FigureCanvas(fig)
    canvas.draw()  # Рендерим график
    graph_image = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)  # Преобразуем в массив
    graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Изменяем форму на (высота, ширина, 4)

    # Преобразуем из ARGB в RGB
    graph_image = np.roll(graph_image, 3, axis=2)  # Перемещаем альфа-канал в конец
    graph_image = graph_image[:, :, :3]  # Убираем альфа-канал

    plt.close(fig)  # Закрываем фигуру
    return graph_image

