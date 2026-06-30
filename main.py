import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import pygame

from PIL import Image, ImageDraw, ImageFont

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize sound system
pygame.mixer.init()

# Load reminder sound
reminder_sound = pygame.mixer.Sound("assets/sounds/reminder.mp3")

# Set sound volume
reminder_sound.set_volume(1.0)

# Webcam
camera = cv2.VideoCapture(0)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink settings
BLINK_THRESHOLD = 0.21
CLOSED_EYES_FRAMES = 3

# Variables
blink_count = 0
frame_counter = 0

# Alert control
alert_played = False

# Calibration
CALIBRATION_TIME = 10

calibration_start = time.time()

calibration_ears = []

is_calibrated = False

# Timers
start_time = time.time()
last_blink_time = time.time()

# Default mode
current_mode = "WORK"
ALERT_SECONDS = 12

# Fonts
title_font = ImageFont.truetype(
    "C:/Windows/Fonts/arialbd.ttf",
    30
)

data_font = ImageFont.truetype(
    "C:/Windows/Fonts/arial.ttf",
    20
)

small_font = ImageFont.truetype(
    "C:/Windows/Fonts/arial.ttf",
    16
)


# Distance function
def euclidean_distance(point1, point2):

    return np.linalg.norm(
        np.array(point1) - np.array(point2)
    )


# EAR calculation
def calculate_ear(eye_points):

    vertical_1 = euclidean_distance(
        eye_points[1],
        eye_points[5]
    )

    vertical_2 = euclidean_distance(
        eye_points[2],
        eye_points[4]
    )

    horizontal = euclidean_distance(
        eye_points[0],
        eye_points[3]
    )

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# Stylish text drawing
def draw_text(frame, text, position, font, color):

    pil_image = Image.fromarray(frame)

    draw = ImageDraw.Draw(pil_image)

    draw.text(
        position,
        text,
        font=font,
        fill=color
    )

    return np.array(pil_image)


while True:

    success, frame = camera.read()

    if not success:
        print("Camera error")
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, _ = frame.shape

    # RGB conversion
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process face mesh
    results = face_mesh.process(rgb_frame)

    # Session timing
    elapsed_time = time.time() - start_time

    elapsed_minutes = elapsed_time / 60

    session_minutes = int(elapsed_time // 60)

    session_seconds = int(elapsed_time % 60)

    # Blink rate
    if elapsed_minutes > 0:

        blink_rate = blink_count / elapsed_minutes

    else:

        blink_rate = 0

    # Face tracking
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            left_eye_points = []
            right_eye_points = []

            # LEFT eye
            for point in LEFT_EYE:

                landmark = face_landmarks.landmark[point]

                x = int(
                    landmark.x * frame_width
                )

                y = int(
                    landmark.y * frame_height
                )

                left_eye_points.append((x, y))

            # RIGHT eye
            for point in RIGHT_EYE:

                landmark = face_landmarks.landmark[point]

                x = int(
                    landmark.x * frame_width
                )

                y = int(
                    landmark.y * frame_height
                )

                right_eye_points.append((x, y))

            # EAR
            left_ear = calculate_ear(
                left_eye_points
            )

            right_ear = calculate_ear(
                right_eye_points
            )

            average_ear = (
                left_ear + right_ear
            ) / 2

            # Blink detection
            if average_ear < BLINK_THRESHOLD:

                frame_counter += 1

            else:

                if frame_counter >= CLOSED_EYES_FRAMES:

                    blink_count += 1

                    last_blink_time = time.time()

                frame_counter = 0

    # Blink inactivity
    time_since_last_blink = (
        time.time() - last_blink_time
    )

    # Status logic
    if elapsed_minutes < 0.5:

        health_status = "CALIBRATING"

        status_color = (255, 220, 0)

    elif time_since_last_blink > ALERT_SECONDS:

        health_status = "BLINK NOW"

        status_color = (255, 80, 80)
        if not alert_played:
         reminder_sound.play()
         alert_played = True

    else:

        health_status = "GOOD"

        status_color = (80, 255, 120)

        alert_played=False

    # Glassmorphism overlay
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (20, 20),
        (420, 290),
        (30, 30, 30),
        -1
    )

    alpha = 0.55

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )

    # App title
    frame = draw_text(
        frame,
        "BlinkWise",
        (40, 35),
        title_font,
        (255, 255, 255)
    )

    # Divider line
    cv2.line(
        frame,
        (40, 80),
        (390, 80),
        (120, 120, 120),
        1
    )

    # Dashboard info
    frame = draw_text(
        frame,
        f"👁  Blinks        : {blink_count}",
        (45, 100),
        data_font,
        (120, 255, 120)
    )

    frame = draw_text(
        frame,
        f"📊  Rate           : {blink_rate:.1f}/min",
        (45, 140),
        data_font,
        (120, 220, 255)
    )

    frame = draw_text(
        frame,
        f"⏱  Session      : {session_minutes:02}:{session_seconds:02}",
        (45, 180),
        data_font,
        (255, 255, 255)
    )

    frame = draw_text(
        frame,
        f"🎮  Mode         : {current_mode}",
        (45, 220),
        data_font,
        (255, 220, 120)
    )

    frame = draw_text(
        frame,
        f"🟢  Status       : {health_status}",
        (45, 260),
        data_font,
        status_color
    )

    # Bottom controls
    frame = draw_text(
        frame,
        "1 Gamer   2 Work   3 Student   Q Exit",
        (25, frame_height - 35),
        small_font,
        (220, 220, 220)
    )

    # Alert banner
    if health_status == "BLINK NOW":

        cv2.rectangle(
            frame,
            (0, frame_height - 100),
            (frame_width, frame_height),
            (20, 20, 180),
            -1
        )

        frame = draw_text(
            frame,
            "⚠ PLEASE BLINK YOUR EYES",
            (70, frame_height - 70),
            title_font,
            (255, 255, 255)
        )

    # Show app
    cv2.imshow(
        "BlinkWise Smart Monitor",
        frame
    )

    # Keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Gamer mode
    elif key == ord('1'):

        current_mode = "GAMER"

        ALERT_SECONDS = 20

    # Work mode
    elif key == ord('2'):

        current_mode = "WORK"

        ALERT_SECONDS = 12

    # Student mode
    elif key == ord('3'):

        current_mode = "STUDENT"

        ALERT_SECONDS = 8

# Cleanup
camera.release()

cv2.destroyAllWindows()