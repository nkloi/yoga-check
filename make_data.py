import glob
import mediapipe as mp
import cv2
import math
import numpy as np
from typing import List, Mapping, Optional, Tuple, Union

OUTPUT_PATH = "data\\data.txt"
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
BG_COLOR = (192, 192, 192)  # gray
LIST_POST = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
             "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
             "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
             "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
             "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", ]


def normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(radians)

    if abs(angle) > 180.0:
        angle = 360 - abs(angle)
    return int(abs(angle))


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    data = [[f, 0] for f in glob.glob('data/forwardBend125/*.*')]
    data = data + [[f, 1] for f in glob.glob('data/forwardBend250/*.*')]
    data = data + [[f, 2] for f in glob.glob('data/forwardBend375/*.*')]
    data = data + [[f, 3] for f in glob.glob('data/forwardBend500/*.*')]
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7) as pose:
        with open(OUTPUT_PATH, "w") as file:
            file.write("angle_1,angle_2,angle_3,angle_4,angle_5,angle_6,angle_7,angle_8,angle_9,"
                       "angle_10,ground_truth,image_path")
            file.write("\n")
            for idx, image_data in enumerate(data):
                for i in range(2):
                    if i == 1:
                        image = cv2.flip(image, 1)
                    image = cv2.imread(image_data[0])
                    image_height, image_width, _ = image.shape
                    # Convert the BGR image to RGB before processing.
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if not results.pose_landmarks:
                        continue
                    print(
                        f'Nose coordinates: ('
                        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
                    )
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)
                        landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    index_left = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                    index_right = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]

                    # Calculate angle
                    angle_1 = calculate_angle(shoulder_right, elbow_right, wrist_right)
                    angle_2 = calculate_angle(shoulder_left, elbow_left, wrist_left)
                    angle_3 = calculate_angle(elbow_right, shoulder_right, hip_right)
                    angle_4 = calculate_angle(elbow_left, shoulder_left, hip_left)
                    angle_5 = calculate_angle(shoulder_right, hip_right, knee_right)
                    angle_6 = calculate_angle(shoulder_left, hip_left, knee_left)
                    angle_7 = calculate_angle(hip_right, knee_right, ankle_right)
                    angle_8 = calculate_angle(hip_left, knee_left, ankle_left)
                    angle_9 = calculate_angle(elbow_left, wrist_left, index_left)
                    angle_10 = calculate_angle(elbow_right, wrist_right, index_right)

                    lic1 = [angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7,
                            angle_8, angle_9, angle_10, image_data[1], image_data[0]]
                    print(lic1)
                    first = True
                    for i in range(len(lic1)):
                        if i != len(lic1) - 1:
                            file.write(str(lic1[i]) + ',')
                        else:
                            file.write(str(lic1[i]))
                    file.write('\n')
