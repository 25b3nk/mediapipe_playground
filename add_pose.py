import cv2

import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


frame_number = 0
csv_data = []


def process_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    return frame


# Open the video file
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_image(frame)
    cv2.namedWindow("MediaPipe Pose", cv2.WINDOW_NORMAL)
    # resized_image = cv2.resize(frame, (800, 800))
    cv2.imshow("MediaPipe Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# path = "/home/bhaskar/Pictures/vishwas.jpg"
# img = cv2.imread(path)
# process_image(img)
