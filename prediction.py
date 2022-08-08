import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

TEST_FILE_PATH = "data/test.txt"

if __name__ == '__main__':

    model = Sequential([
        Dense(16, activation='relu', name="L1"),
        Dense(4, activation='linear', name="L2")
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    model = load_model('forwardBend_best.h5')

    # FOR TEST
    data = pd.read_csv(TEST_FILE_PATH)
    X_test = data.drop(columns=['ground_truth', 'image_path'])
    image_path = data['image_path']

    prediction = model.predict(X_test)
    prediction_p = tf.nn.softmax(prediction)
    score = [125, 250, 375, 500]
    for index, item in enumerate(prediction_p):
        yhat = np.argmax(item)
        print(image_path[index], item, score[yhat])
