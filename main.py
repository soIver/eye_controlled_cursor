import cv2
import mediapipe as mp
import pyautogui

# Инициализация видеозахвата с веб-камеры
cam = cv2.VideoCapture(0)
# Инициализация MediaPipe Face Mesh для обнаружения лицевых меток
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# Получение размеров экрана для преобразования координат
screen_w, screen_h = pyautogui.size()

# Основной цикл обработки видео
while True:
    # Чтение кадра с камеры
    _, frame = cam.read()
    # Зеркальное отражение кадра для естественного восприятия
    frame = cv2.flip(frame, 1)
    # Конвертация цветового пространства из BGR (OpenCV) в RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Обработка кадра для обнаружения лицевых меток
    output = face_mesh.process(rgb_frame)
    # Получение точек меток
    landmark_points = output.multi_face_landmarks
    # Получение размеров кадра
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        # Обработка точек глаз (метки 474-478 - область вокруг зрачка)
        for id, landmark in enumerate(landmarks[474:478]):
            # Преобразование нормализованных координат в пиксельные
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # Рисование зеленых кругов на точках глаз
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            # Для центральной точки зрачка
            if id == 1:
                # Преобразование координат в координаты экрана
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                # Перемещение курсора мыши
                pyautogui.moveTo(screen_x, screen_y)

        # Определение точек для детекции моргания (верхнее и нижнее веко левого глаза)
        left = [landmarks[145], landmarks[159]]
        # Рисование желтых кругов на точках век
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        
        # Детекция моргания по расстоянию между веками
        if (left[0].y - left[1].y) < 0.004:
            # Выполнение клика мыши
            pyautogui.click()
            # Пауза для предотвращения множественных кликов
            pyautogui.sleep(1)
    
    # Отображение окна с обработанным видео
    cv2.imshow('Eye Controlled Mouse', frame)
    # Задержка в 1 мс перед следующей итерацией цикла
    cv2.waitKey(1)