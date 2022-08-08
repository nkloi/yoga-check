from cgitb import lookup
from telnetlib import FORWARD_X
from turtle import forward
import cv2
from matplotlib import image
import mediapipe as mp
import numpy as np
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
# For static images:
IMAGE_FILES = []

STANDARD_ANGLE_POSE = []
# T_POSE = {'elbow_right' : [165, 175], 'elbow_left' : [165, 175], 'shoulder_right' : [85, 105], 'shoulder_left' : [85, 105], 'hip_right' :[170, 180], 'hip_left' : [170, 180], 'knee_right' : [170, 180], 'knee_left' : [170, 180]}
T_POSE = [[160, 175], [160, 175], [70, 105], [70, 105], [170, 180], [170, 180], [170, 180], [170, 180]]

T_POSE_1 = [[160, 170], [160, 180], [80, 105], [80, 105], [170, 180], [170, 180], [170, 180], [170, 180]]
T_POSE_2 = [[170, 180], [160, 180], [70, 90], [70, 90], [170, 180], [170, 180], [170, 180], [170, 180]]
T_POSE_3 = [[160, 180], [160, 180], [60, 80], [60, 80], [170, 180], [170, 180], [170, 180], [170, 180]]
T_POSE_4 = [[160, 180], [160, 180], [50, 70], [50, 70], [170, 180], [170, 180], [170, 180], [170, 180]]
FORWARD_BAND_LIST = [T_POSE_1, T_POSE_2, T_POSE_3, T_POSE_4]

# FORWARD_BAND_1 = [[160, 180], [160, 180],[100, 170], [100, 170], [15, 30],[15, 30], [160, 180], [160, 180]]
FORWARD_BAND_1 = [[160, 170], [160, 170], [130, 140], [130, 140], [15, 25], [15, 25], [165, 180], [170, 185]]
FORWARD_BAND_2 = [[170, 180], [165, 175], [140, 160], [140, 160], [15, 30], [15, 30], [170, 185], [170, 185]]
FORWARD_BAND_3 = [[160, 180], [170, 180], [105, 120], [105, 120], [45, 55], [40, 55], [160, 175], [155, 170]]
FORWARD_BAND_4 = [[160, 170], [160, 170], [50, 70], [50, 70], [70, 90], [70, 90], [170, 185], [165, 180]]
FORWARD_BAND_LIST = [FORWARD_BAND_1, FORWARD_BAND_2, FORWARD_BAND_3, FORWARD_BAND_4]

# IMAGE_FILES.append('image\\forwardBend125.jpg')
# IMAGE_FILES.append('image\\forwardBend200.jpg')
# IMAGE_FILES.append('image\\forwardBend375.jpg')
# IMAGE_FILES.append('image\\forwardBend500.jpg')

for i in range(60):
    address_image = 'data\\forwardBend500\\' + str(i + 1) + '.' + 'jpg'
    IMAGE_FILES.append([address_image, 3])

for i in range(60):
    address_image = 'data\\forwardBend375\\' + str(i + 1) + '.' + 'jpg'
    IMAGE_FILES.append([address_image, 2])

for i in range(60):
    address_image = 'data\\forwardBend250\\' + str(i + 1) + '.' + 'jpg'
    IMAGE_FILES.append([address_image, 1])

for i in range(60):
    address_image = 'data\\forwardBend125\\' + str(i + 1) + '.' + 'jpg'
    IMAGE_FILES.append([address_image, 0])

# IMAGE_FILES.append('data\\forwardBend375\\19.jpg')

BG_COLOR = (192, 192, 192)  # gray
LIST_POST = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
             "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
             "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
             "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
             "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", ]


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(radians)

    if abs(angle) > 180.0:
        angle = 360 - abs(angle)
    return abs(angle)


def is_pose_in_POSTLIST(list_angle, POSE):
    print("===============")
    print(list_angle, POSE)
    is_pose_in = True
    for i in range(len(list_angle)):
        if list_angle[i] > POSE[i][0] and list_angle[i] < POSE[i][1]:
            continue
        # print(i) #determine number incorrect coordinates
        is_pose_in = False
        break
    return is_pose_in


def pose_detection(list_angle, POSE_LIST):
    mask_list = [500, 375, 200, 125]
    # is_yoga_pose = "FORWARD BEND"
    is_yoga_pose = "T POSE"
    mask = 125
    color = (0, 255, 0)
    # print(POSE_LIST[0][0]) #debug - test
    # print(list_angle[0])
    for i in range(len(POSE_LIST)):
        if is_pose_in_POSTLIST(list_angle, POSE_LIST[i]):
            # is_yoga_pose = "FORWARD BEND"
            is_yoga_pose = "T POSE"
            mask = mask_list[i]
            color = (0, 255, 0)
            break
        else:
            # is_yoga_pose = "Not FORWARD BEND"
            is_yoga_pose = "Not T POSE"
    if is_yoga_pose == "Not T POSE":
        color = (0, 0, 255)
        mask = 0
    return is_yoga_pose, color, mask


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
    return (x_px, y_px)


def convert_str2list(image_list):
    list_angle = []
    for i in range(8):
        index_space = image_list.index(" ")
        list_angle.append(int(image_list[0:index_space]))
        image_list = image_list[index_space + 1:]
    return list_angle


forwardBend_angle_list = []

# with open("data\data.txt", "r") as file:

# 	for i in range(240):
# 		image_list = file.readline()
# 		list_angle = convert_str2list(image_list)
# 		forwardBend_angle_list.append(list_angle)

# print(len(forwardBend_angle_list))
# forwardBend_angle_list = np.array(forwardBend_angle_list)

# for i in range(len(forwardBend_angle_list)):
# 	print(forwardBend_angle_list[i])

if __name__ == '__main__':
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        with open("data\data.txt", "a") as file:
            file.write("angle_1,angle_2,angle_3,angle_4,angle_5,angle_6,angle_7,angle_8,angle_9,angle_10,ground_truth")
            file.write("\n")
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file[0])
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

            lic1 = [angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7, angle_8, angle_9, angle_10, file[1]]
            print(lic1)
            forwardBend_angle_list.append(lic1)

            with open("data\data.txt", "a") as file:
                for i in range(len(lic1)):
                    if i < len(lic1) - 1:
                        file.write(str(round(lic1[i])) + ",")
                    else:
                        file.write(str(round(lic1[i])))
                file.write("\n")
