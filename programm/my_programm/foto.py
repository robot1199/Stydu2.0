import cv2

# Загрузка предобученного классификатора для лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения
image = cv2.imread('pikcher.jpg')

# Преобразование в оттенки серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Рисование прямоугольников вокруг лиц
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Отображение результата
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()