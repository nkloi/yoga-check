import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
def convert_str2list(image_list):
	list_angle = []
	for i in range(8):
		index_space = image_list.index(" ")
		list_angle.append(int(image_list[0:index_space]))
		image_list = image_list[index_space + 1: ]
	return list_angle

forwardBend_angle_list = []

with open("data/data.txt") as file:
	for i in range(240):
		image_list = file.readline()
		list_angle = convert_str2list(image_list)
		forwardBend_angle_list.append(list_angle)

print(len(forwardBend_angle_list))
forwardBend_angle_list = np.array(forwardBend_angle_list)

X_train= forwardBend_angle_list
y_train=  np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
                    ,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
y_train = y_train.reshape(-1,1)
print(X_train.shape)
print(y_train.shape)
tf.random.set_seed(1234)
model = Sequential(
        [
            Dense(8, activation = 'relu',   name = "L1"),
            Dense(4, activation = 'linear', name = "L2")
        ]
    )
model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.03),
    )

model.fit(
        X_train,y_train,
        epochs=1000
    )
X_test= np.array([174, 180, 54, 55, 82, 78, 170, 170])
prediction = model.predict(X_test)
prediction_p = tf.nn.softmax(prediction)
yhat = np.argmax(prediction_p)
print(prediction_p )
print(yhat)