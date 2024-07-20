import sys

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Drawing utilities
fm = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

frame_number = 0
csv_data = []
moustache_path = "moustache.png"
moustache_img = cv2.imread("moustache.png", cv2.IMREAD_UNCHANGED)


def apply_filter(image, face_landmarks):
    image_width = image.shape[1]
    image_height = image.shape[0]
    # upper_lip_inner_corner = face_landmarks.landmark[78]
    upper_lip_outer_corner = face_landmarks.landmark[14]
    nose_tip = face_landmarks.landmark[1]
    left_lip_tip = face_landmarks.landmark[23]
    right_lip_tip = face_landmarks.landmark[21]
    moustache_width = int(abs(left_lip_tip.x - right_lip_tip.x) * image_width)
    moustache_height = int(abs(nose_tip.y - upper_lip_outer_corner.y) * image_height)
    print(moustache_width, moustache_height)
    # # Resize the moustache image to match the calculated size
    moustache_resized = cv2.resize(moustache_img, (moustache_width, moustache_height))
    print(moustache_resized.shape)
    # Create a mask for the moustache
    mask = moustache_resized[:, :, :3] / 255.0
    m_points = np.float32([[158, 49], [2, 87], [318, 87]])
    p_points = np.float32(
        [
            [nose_tip.x * image_width, nose_tip.y * image_height],
            [left_lip_tip.x * image_width, left_lip_tip.y * image_height],
            [right_lip_tip.x * image_width, right_lip_tip.y * image_height],
        ]
    )
    transform = cv2.getAffineTransform(m_points, p_points)

    # Warp the moustache to fit the face region based on calculated transformation matrix
    x = int(min(left_lip_tip.x, right_lip_tip.x) * image_width)
    y = int(min(nose_tip.y, upper_lip_outer_corner.y) * image_height)
    warped_moustache = cv2.warpAffine(
        moustache_resized[:, :, :3],
        transform,
        (
            image[y : y + moustache_height, x : x + moustache_width].shape[1],
            image[y : y + moustache_height, x : x + moustache_width].shape[0],
        ),
    )

    # print(x, x + moustache_width, y, y + moustache_height)
    # print(image.shape)
    # print(mask.shape)
    # print((1 - mask).shape)
    # print(moustache_resized[:, :, :3].shape)
    # # print(warped_moustache.shape)
    # print(image[y : y + moustache_height, x : x + moustache_width].shape)
    # Apply the moustache to the image using the mask
    image[y : y + moustache_height, x : x + moustache_width] = (1 - mask) * image[
        y : y + moustache_height, x : x + moustache_width
    ] + mask * warped_moustache
    return image


def process_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Pose
    result = fm.process(frame_rgb)
    lm_present = False
    # Draw the pose landmarks on the frame
    if result.multi_face_landmarks:
        lm_present = True
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=result.multi_face_landmarks[0],
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        frame = apply_filter(frame, result.multi_face_landmarks[0])
    return frame, lm_present


# Open the video file
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame, lm_present = process_image(frame)
    # if not lm_present:
    #     continue
    # cv2.imwrite("output.png", frame)
    # sys.exit(0)
    cv2.namedWindow("MediaPipe Pose", cv2.WINDOW_NORMAL)
    # resized_image = cv2.resize(frame, (800, 800))
    cv2.imshow("MediaPipe Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
