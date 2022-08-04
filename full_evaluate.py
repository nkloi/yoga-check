from re import X
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.models import load_model

angle_predict= np.array([[170, 175, 69 ,70, 74 ,73, 179, 177 ]])

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

# print(angle_accuracy_list[0][0])
evaluate_yogapose(weight_file, angle_accuracy_list,  angle_predict)