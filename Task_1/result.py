import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def test():
    print("data loaded")
    data = pd.read_csv("X_exam.csv")

    # Encode the data
    data_transformed = data.loc[:, ['gender', 'age_code', 'region_code']]
    data_transformed.head()

    print("Converting to Weekly data")
    for i in range(3):
        j = 238 * i + 3
        k = 1
        while j < 238 * (i + 1) + 3:
            weekly_info = np.sum(data.iloc[:, j:j + 7], axis=1) / 7
            if i == 0:
                data_transformed[f'c_week{k}'] = weekly_info
            elif i == 1:
                data_transformed[f's_week{k}'] = weekly_info
            else:
                data_transformed[f't_week{k}'] = weekly_info
            k += 1
            j += 7

    one_hot = pd.get_dummies(data_transformed, columns=['gender', 'age_code', 'region_code'])
    one_hot.head()
    print("Data converted to weekly data and encoded")

    del data_transformed
    del data
    print("Data deleted")

    print("Loading model and scaler")
    # Scale the data
    scaler = pickle.load(open("scaler.obj", "rb"))
    one_hot = scaler.transform(one_hot)

    # Load the models
    NN_models = pickle.load(open("./NNModels.obj", "rb"))

    # Predict the data
    print("Predicting")
    predictions = []
    for model in NN_models:
        predictions.append(model.predict(one_hot))

    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)

    # Save the predictions
    np.savetxt("predictions.csv", predictions, delimiter=",")
    print("Predictions saved to predictions.csv")

    return predictions

if __name__ == "__main__":
    test()


