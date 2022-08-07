from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DATA_FILE_PATH = "data/data.txt"
TEST_FILE_PATH = "data/test.txt"


if __name__ == '__main__':
    data = pd.read_csv(DATA_FILE_PATH)
    X_data = data.drop(columns=['ground_truth', 'image_path'])
    y_data = data['ground_truth']
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=2022)

    clf = MLPClassifier(hidden_layer_sizes=(16, 4), random_state=1, max_iter=1, warm_start=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Accuracy of 1NN for YOGA: %.2f %%" % (100 * accuracy_score(y_val, y_pred)))

    # FOR TEST
    data = pd.read_csv(TEST_FILE_PATH)
    X_test = data.drop(columns=['ground_truth', 'image_path'])
    image_path = data['image_path']

    # X_test = np.array([[174, 174, 120, 122, 39, 38, 171, 173, 171, 176],  # 250
    #                    [163, 162, 109, 106, 42, 43, 179, 178, 173, 176],  # 375
    #                    [167, 158, 32, 21, 99, 99, 168, 167, 163, 176],  # 125
    #                    [146, 142, 25, 24, 83, 82, 15, 12, 155, 159]])  # Your yoga pose incorrect

    prediction = clf.predict(X_test)
    print(clf.predict_proba(X_test))
    score = [125, 250, 375, 500]
    for index, item in enumerate(prediction):
        print(image_path[index], score[item])
