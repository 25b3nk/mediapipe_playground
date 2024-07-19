import cv2

import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
fm = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


frame_number = 0
csv_data = []


def process_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Pose
    result = fm.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=result.multi_face_landmarks[0],
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
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
