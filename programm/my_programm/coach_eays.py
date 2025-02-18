import cv2

# Открытие видеофайла
cap = cv2.VideoCapture('test_video.mp4')
frames = []

# Сохранение всех кадров в список
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# Переменные для управления воспроизведения
current_frame = 0
playback_speed = 100 # задержка в милисекундах(100 мс = нормальная скорость)
is_playing = True


while True:
    if is_playing:
        # Отображение текущего кадра
        cv2.imshow('Video', frames[current_frame])
        current_frame = (current_frame + 1) % len((frames)) # циклический переход к началу

    key = cv2.waitKey(playback_speed)    # Задержка для воспроизведения
    if key & 0xFF == ord('q'):     # Выход
        break
    elif key & 0xFF == ord('s'): # Остановка
        is_playing = not is_playing
        print('stop/go')
    elif key & 0xFF == ord('b'): # возврат на один кадр
        current_frame = max(0, current_frame - 1)
    elif key & 0xFF == ord('f'): # вперед на один кадр
        current_frame = min(len(frames) -1, current_frame - 1)
        print(playback_speed)
    elif key & 0xFF == ord('+'): # Ускорение
        playback_speed = max(1, playback_speed - 10) # Ускорение задержки
        print(playback_speed)
    elif key & 0xFF == ord('-'): # Замедление
        playback_speed += 10 # Увиличение задкржки





# Освобождение ресурсов
cv2.destroyAllWindows()