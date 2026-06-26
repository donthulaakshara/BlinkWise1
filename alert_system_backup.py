import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
camera = cv2.VideoCapture(0)

# Eye landmark points
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink settings
BLINK_THRESHOLD = 0.21
CLOSED_EYES_FRAMES = 3

# Variables
blink_count = 0
frame_counter = 0

# Timers
start_time = time.time()
last_blink_time = time.time()

# Alert settings
ALERT_SECONDS = 10


# Distance calculation
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# EAR calculation
def calculate_ear(eye_points):

    vertical_1 = euclidean_distance(eye_points[1], eye_points[5])
    vertical_2 = euclidean_distance(eye_points[2], eye_points[4])

    horizontal = euclidean_distance(eye_points[0], eye_points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


while True:

    success, frame = camera.read()

    if not success:
        print("Failed to capture frame.")
        break

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # Frame size
    frame_height, frame_width, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
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

    # Detect face
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            left_eye_points = []
            right_eye_points = []

            # LEFT eye
            for point in LEFT_EYE:

                landmark = face_landmarks.landmark[point]

                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                left_eye_points.append((x, y))

            # RIGHT eye
            for point in RIGHT_EYE:

                landmark = face_landmarks.landmark[point]

                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                right_eye_points.append((x, y))

            # EAR calculation
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            average_ear = (left_ear + right_ear) / 2

            # Blink detection
            if average_ear < BLINK_THRESHOLD:

                frame_counter += 1

            else:

                if frame_counter >= CLOSED_EYES_FRAMES:
                    blink_count += 1
                    last_blink_time = time.time()

                frame_counter = 0

    # Time since last blink
    time_since_last_blink = time.time() - last_blink_time

    # Health status
    if elapsed_minutes < 0.5:

        health_status = "CALIBRATING"
        health_color = (0, 255, 255)

    elif time_since_last_blink > ALERT_SECONDS:

        health_status = "BLINK NOW"
        health_color = (0, 0, 255)

    else:

        health_status = "GOOD"
        health_color = (0, 255, 0)

    # Overlay panel
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (470, 260),
        (0, 0, 0),
        -1
    )

    alpha = 0.5

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )

    # Title
    cv2.putText(
        frame,
        "BlinkWise Smart Monitor",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Blink count
    cv2.putText(
        frame,
        f"Total Blinks: {blink_count}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Blink rate
    cv2.putText(
        frame,
        f"Blink Rate: {blink_rate:.1f}/min",
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Session timer
    cv2.putText(
        frame,
        f"Session: {session_minutes:02}:{session_seconds:02}",
        (25, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Status
    cv2.putText(
        frame,
        f"Status: {health_status}",
        (25, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        health_color,
        2
    )

    # BIG ALERT BANNER
    if health_status == "BLINK NOW":

        cv2.rectangle(
            frame,
            (0, frame_height - 100),
            (frame_width, frame_height),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "BLINK YOUR EYES NOW!",
            (50, frame_height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

    # Show app
    cv2.imshow("BlinkWise", frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()