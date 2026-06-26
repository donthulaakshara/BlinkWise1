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

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink settings
BLINK_THRESHOLD = 0.21
CLOSED_EYES_FRAMES = 3

# Variables
blink_count = 0
frame_counter = 0

# Timer
start_time = time.time()

# Warning settings
LOW_BLINK_THRESHOLD = 10


# Distance function
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

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Frame size
    frame_height, frame_width, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Convert to minutes
    elapsed_minutes = elapsed_time / 60

    # Avoid division by zero
    if elapsed_minutes > 0:
        blink_rate = blink_count / elapsed_minutes
    else:
        blink_rate = 0

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

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # RIGHT eye
            for point in RIGHT_EYE:

                landmark = face_landmarks.landmark[point]

                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                right_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Calculate EAR
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)

            average_ear = (left_ear + right_ear) / 2

            # Blink logic
            if average_ear < BLINK_THRESHOLD:

                frame_counter += 1

            else:

                if frame_counter >= CLOSED_EYES_FRAMES:
                    blink_count += 1

                frame_counter = 0

            # Display blink count
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display blink rate
            cv2.putText(
                frame,
                f"Blinks/Min: {blink_rate:.1f}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display EAR
            cv2.putText(
                frame,
                f"EAR: {average_ear:.2f}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Low blink warning
            if blink_rate < LOW_BLINK_THRESHOLD and elapsed_minutes > 0.5:

                cv2.putText(
                    frame,
                    "WARNING: Blink More!",
                    (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # Show frame
    cv2.imshow("BlinkWise Smart Monitor", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()