
from re import X
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.models import load_model



def convert_str2list(image_list):
	list_angle = []
	for i in range(10):
		index_space = image_list.index(" ")
		list_angle.append(int(image_list[0:index_space]))
		image_list = image_list[index_space + 1: ]
	return list_angle

forwardBend_angle_list = []
forwardBend_angle_list_val = []
with open(r"data\data.txt") as file:
	for i in range(240):
		image_list = file.readline()
		list_angle = convert_str2list(image_list)
		forwardBend_angle_list.append(list_angle)

print(len(forwardBend_angle_list))
forwardBend_angle_list = np.array(forwardBend_angle_list)
X_train = forwardBend_angle_list
y_train = np.zeros(240)
y_train[0:60] = 3*np.ones(60)
y_train[60:120] = 2*np.ones(60)
y_train[120:180] = 1*np.ones(60)
y_train = y_train.astype('uint8')
y_train = y_train.reshape(-1,1)

with open(r"data\data.txt") as file:
	
	for i in range(240):
		image_list1 = file.readline()
		list_angle1 = convert_str2list(image_list1)
		forwardBend_angle_list_val.append(list_angle1)


print(len(forwardBend_angle_list_val))
forwardBend_angle_list_val= np.array(forwardBend_angle_list_val)
y_train = y_train.reshape(-1,1)
X_val =forwardBend_angle_list_val
y_val = np.zeros(240)
y_val[0:60] = 3*np.ones(60)
y_val[60:120] = 2*np.ones(60)
y_val[120:180] = 1*np.ones(60)
y_val = y_val.astype('uint8')
y_val = y_val.reshape(-1,1)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)

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

model.fit(
        X_train,y_train,validation_data = (X_val, y_val),
        epochs=1000
    )
model.save('viet2.h5')
model = load_model('viet2.h5')
import numpy as np
X_test= np.array([[170, 175, 69 ,70, 74 ,73, 179, 177, 120, 120 ]])

prediction = model.predict(X_test)
prediction_p = tf.nn.softmax(prediction)
yhat = np.argmax(prediction_p)
print(prediction_p )
print((yhat+1)*125)
