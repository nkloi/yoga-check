from telnetlib import FORWARD_X
from turtle import forward
import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Mapping, Optional, Tuple, Union
import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from re import X
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5
# For static images:
IMAGE_FILES = []

STANDARD_ANGLE_POSE = []
# T_POSE = {'elbow_right' : [165, 175], 'elbow_left' : [165, 175], 'shoulder_right' : [85, 105], 'shoulder_left' : [85, 105], 'hip_right' :[170, 180], 'hip_left' : [170, 180], 'knee_right' : [170, 180], 'knee_left' : [170, 180]}
T_POSE = [[160, 175], [160, 175],[70, 105], [70, 105], [170, 180],[170, 180], [170, 180], [170, 180]]

T_POSE_1 = [[160, 170], [160, 180],[80, 105], [80, 105], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_2 = [[170, 180], [160, 180],[70, 90], [70, 90], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_3 = [[160, 180], [160, 180],[60, 80], [60, 80], [170, 180],[170, 180], [170, 180], [170, 180]]
T_POSE_4 = [[160, 180], [160, 180],[50, 70], [50, 70], [170, 180],[170, 180], [170, 180], [170, 180]]
FORWARD_BAND_LIST = [T_POSE_1, T_POSE_2, T_POSE_3, T_POSE_4]

# FORWARD_BAND_1 = [[160, 180], [160, 180],[100, 170], [100, 170], [15, 30],[15, 30], [160, 180], [160, 180]]
FORWARD_BAND_1 = [[160, 170], [160, 170],[130, 140], [130, 140], [15, 25],[15, 25], [165, 180], [170, 185]]
FORWARD_BAND_2 = [[170, 180], [165, 175],[140, 160], [140, 160], [15, 30],[15, 30], [170, 185], [170, 185]]
FORWARD_BAND_3 = [[160, 180], [170, 180],[105, 120], [105, 120], [45, 55],[40, 55], [160, 175], [155, 170]]
FORWARD_BAND_4 = [[160, 170], [160, 170],[50, 70], [50, 70], [70, 90],[70, 90], [170, 185], [165, 180]]
FORWARD_BAND_LIST = [FORWARD_BAND_1, FORWARD_BAND_2, FORWARD_BAND_3, FORWARD_BAND_4]


# IMAGE_FILES.append('image\\forwardBend125.jpg')
# IMAGE_FILES.append('image\\forwardBend200.jpg')
# IMAGE_FILES.append('image\\forwardBend375.jpg')
# IMAGE_FILES.append('image\\forwardBend500.jpg')

# address_image = 'data\\forwardBend500\\' + str(1) + '.' + 'jpg'
# for i in range(30):
#     address_image = 'data\\forwardBend500\\' + str(i+1) + '.' + 'jpg'
#     IMAGE_FILES.append(address_image)

# for i in range(30):
#     address_image = 'data\\forwardBend375\\' + str(i+1) + '.' + 'jpg'
#     IMAGE_FILES.append(address_image)


IMAGE_FILES.append('image\\forwardBend500.jpg')


BG_COLOR = (192, 192, 192)  # gray
LIST_POST = ["NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
             "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
             "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
             "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
             "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", ]

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.rad2deg(radians)
    
    if abs(angle) > 180.0:
        angle = 360-abs(angle)
    return abs(angle) 

def is_pose_in_POSTLIST(list_angle, POSE):
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


# angle_predict= np.array([[170, 175, 69 ,70, 74 ,73, 179, 177 ]])

angle_accuracy_500 = np.array([161, 161, 115, 115, 32, 32, 174, 176])
angle_accuracy_375 = np.array([166, 164, 121, 119, 44, 43, 173, 173])
angle_accuracy_200 = np.array([173, 172, 113, 112, 60, 60, 175, 175])
angle_accuracy_125 = np.array([167, 166, 54, 54, 79, 79, 175, 175])

angle_accuracy_list = [angle_accuracy_125, angle_accuracy_200, angle_accuracy_375, angle_accuracy_500]
weight_file = 'viet.h5'

def score_mask_class(weight_file, angle_predict):
	tf.random.set_seed(1234)
	model = Sequential(
	        [
	            Dense(16, activation = 'relu',   name = "L1"),
	            Dense(4, activation = 'linear', name = "L2")
	        ]
	    )
	model.compile(
	        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	        optimizer=tf.keras.optimizers.Adam(0.01),
	    )

	model = load_model(weight_file)
	prediction = model.predict(angle_predict)
	prediction_p = tf.nn.softmax(prediction)
	mask_class= np.argmax(prediction_p)  + 1
	mask = (mask_class)*125
	return mask_class, mask

def cal_accuracy(angle_accuracy, angle_predict):
	err = np.abs(angle_predict - angle_accuracy)
	accuracy = 1 - err/angle_accuracy
	accuracy_average = np.sum(accuracy)/8
	return accuracy_average

def create_comment(angle_accuracy, angle_predict):
	list_position = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	list_angle = ["elbow_right", "elbow_left", "shoulder_right","shoulder_left", "hip_right", "hip_left", "knee_right", "knee_left" ]

	is_correct = True
	comment = "Warning  : "

	# Determine - incorrect postion
	for i in range(len(angle_accuracy)):
		if (angle_predict[0][i] < (angle_accuracy[i] - 10)) or (angle_predict[0][i] > (angle_accuracy[i] + 10)) :
			list_position[i] = 1
			comment += list_angle[i] + " ,"
			is_correct = False

	comment = comment[:(len(comment) - 1)]
	if is_correct:
		comment = "Wonderful Pose Yoga"
	return comment
	
def evaluate_yogapose(weight_file, angle_accuracy_list,  angle_predict):
	mask_class, mask = score_mask_class(weight_file, angle_predict)
	print(mask_class, mask)
	angle_accuracy = np.zeros(8)

	for i in range(4):
		if mask_class == (i + 1):
			angle_accuracy = angle_accuracy_list[i]
			break
	accuracy_average = cal_accuracy(angle_accuracy, angle_predict)
	if accuracy_average > 0.6:
		comment = create_comment(angle_accuracy, angle_predict)
	else:
		comment = "Your yoga pose incorrect"

	print("Score : ", mask)
	print("Accuracy is : {} %".format(round(accuracy_average * 100,2)))
	print(comment)
	return None

# evaluate_yogapose(weight_file, angle_accuracy_list,  angle_predict)

forwardBend_angle_list = []

if __name__ == '__main__':
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
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
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # Calculate angle
            angle_1 = calculate_angle(shoulder_right, elbow_right, wrist_right)
            angle_2 = calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_3 = calculate_angle(elbow_right, shoulder_right, hip_right)
            angle_4 = calculate_angle(elbow_left, shoulder_left,hip_left)
            angle_5 = calculate_angle(shoulder_right, hip_right, knee_right)
            angle_6 = calculate_angle(shoulder_left, hip_left, knee_left)
            angle_7 = calculate_angle(hip_right, knee_right, ankle_right)
            angle_8 = calculate_angle(hip_left, knee_left, ankle_left)
            
            lic1 = [angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7, angle_8]
            print(lic1)
            forwardBend_angle_list.append(lic1)

            # with open("data\data.txt", "a") as file:
            #     for i in range(len(lic1)):
            #         file.write(str(round(lic1[i])) + " ")
            #     file.write("\n")
            angle_predict= np.array([lic1])
            evaluate_yogapose(weight_file, angle_accuracy_list,  angle_predict)


            # is_yoga_pose, color, mask = pose_detection(lic1, FORWARD_BAND_LIST)
            # # print(is_yoga_pose)

            # Print Pose_detection
            # cv2.putText(image, is_yoga_pose + str(": ") + str(mask), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # print(results.segmentation_mask)
            
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose',image)

            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
            # # # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # For webcam input:
    # cap = cv2.VideoCapture(0)
    # with mp_pose.Pose(
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5) as pose:
    #     while cap.isOpened():
    #         success, image = cap.read()
    #         if not success:
    #             print("Ignoring empty camera frame.")
    #             # If loading a video, use 'break' instead of 'continue'.
    #             continue
    #
    #         # To improve performance, optionally mark the image as not writeable to
    #         # pass by reference.
    #         image.flags.writeable = False
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         image = cv2.flip(image, 1)
    #         results = pose.process(image)
    #         image_rows, image_cols, _ = image.shape
    #
    #         lic = {'angle_1': 'elbow_right', 'angle_2': 'elbow_left', 'angle_3': 'shoulder_right', 'angle_4': 'shoulder_left', 'angle_5': 'hip_right', 'angle_6': 'hip_left', 'angle_7': 'knee_right', 'angle_8':'knee_left'}
    #         for idx, landmark in enumerate(results.pose_landmarks.landmark):
    #             idx_to_coordinates = {}
    #             if ((landmark.HasField('visibility') and
    #                  landmark.visibility < VISIBILITY_THRESHOLD) or
    #                     (landmark.HasField('presence') and
    #                      landmark.presence < PRESENCE_THRESHOLD)):
    #                 continue
    #
    #             landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
    #             landmarks = results.pose_landmarks.landmark
    #
    #             # Get coordinates
    #             shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    #             elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    #             wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    #             shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    #             hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    #             ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    #             hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    #             ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    #             knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    #             knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    #             elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    #             wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    #             # Calculate angle
    #             angle_1 = calculate_angle(shoulder_right, elbow_right, wrist_right)
    #             angle_2 = calculate_angle(shoulder_left, elbow_left, wrist_left)
    #             angle_3 = calculate_angle(elbow_right, shoulder_right, hip_right)
    #             angle_4 = calculate_angle(elbow_left, shoulder_left,hip_left)
    #             angle_5 = calculate_angle(shoulder_right, hip_right, knee_right)
    #             angle_6 = calculate_angle(shoulder_left, hip_left, knee_left)
    #             angle_7 = calculate_angle(hip_right, knee_right, ankle_right)
    #             angle_8 = calculate_angle(hip_left, knee_left, ankle_left)
    #
    #             lic1 = [angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7, angle_8]
    #             lic2 = [elbow_right, elbow_left, shoulder_right,shoulder_left, hip_right, hip_left, knee_right, knee_left ]
    #             # Visualize angle
    #             # lic = round([angle_1, angle_2, angle_3, angle_4, angle_5,angle_6,angle_7,angle_8],2)
    #             # lic = {'angle_1': 'elbow_right', 'angle_2': 'elbow_left', 'angle_3': 'shoulder_right', 'angle_4': 'shoulder_left', 'angle_5': 'hip_right', 'angle_6': 'hip_left', 'angle_7': 'knee_right', 'angle_8':'knee_left'}
    #             print(lic1)
    #             is_yoga_pose, color, mask = pose_detection(lic1, FORWARD_BAND_LIST)
    #             print(is_yoga_pose)
    #             cv2.putText(image, is_yoga_pose + str(": ") + str(mask), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    #             for i in range(len(lic1)):
    #                 cv2.putText(image, str(round(lic1[i],2)),
    #                             tuple(np.multiply(lic2[i], [640, 480]).astype(int)),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    #                                     )
    #         # Draw the pose annotation on the image.
    #         image.flags.writeable = True
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         mp_drawing.draw_landmarks(
    #             image,
    #             results.pose_landmarks,
    #             mp_pose.POSE_CONNECTIONS,
    #             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #         # Flip the image horizontally for a selfie-view display.
    #         cv2.imshow('MediaPipe Pose',image)
    #         if cv2.waitKey(5) & 0xFF == ord('q'):
    #             break
    # cap.release()
