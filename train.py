import glob

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

DATA_FILE_PATH = "data/data.txt"
TEST_FILE_PATH = "data/test.txt"


if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE_PATH)
    X_data = data.drop(columns=['ground_truth','image_path'])
    y_data = data['ground_truth']
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=2022)

    model = Sequential([
        Dense(16, activation='relu', name="L1"),
        Dense(4, activation='linear', name="L2")
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True, patience=20)

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=1000,
        callbacks=[es]
    )

    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('forwardBend.h5')
    # model = load_model('forwardBend.h5')

    # FOR TEST
    data = pd.read_csv(TEST_FILE_PATH)
    X_test = data.drop(columns=['ground_truth', 'image_path'])
    image_path = data['image_path']

    # X_test = np.array([[174, 174, 120, 122, 39, 38, 171, 173, 171, 176],  # 250
    #                    [163, 162, 109, 106, 42, 43, 179, 178, 173, 176],  # 375
    #                    [167, 158, 32, 21, 99, 99, 168, 167, 163, 176],  # 125
    #                    [146, 142, 25, 24, 83, 82, 15, 12, 155, 159]])  # Your yoga pose incorrect

    prediction = model.predict(X_test)
    prediction_p = tf.nn.softmax(prediction)
    score = [125, 250, 375, 500]
    for index, item in enumerate(prediction_p):
        yhat = np.argmax(item)
        print(image_path[index], item, score[yhat])
