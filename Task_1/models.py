import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
# from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep
from tabtransformertf.utils.preprocessing import build_numerical_prep

import tensorflow_addons as tfa
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import average_precision_score, roc_auc_score

# path = "/content/drive/MyDrive/Colab Notebooks/College/data-science comp/"
path = ""

# Read the data
train_data = pd.read_pickle(path + "train_task1.pkl")
test_data = pd.read_pickle(path + "test_task1.pkl")

train_data["total logins"] = 0
test_data["total logins"] = 0
train_data["total logins with transfer"] = 0
test_data["total logins with transfer"] = 0
train_data["total logins duration"] = 0
test_data["total logins duration"] = 0

for col in train_data.columns:
    if "c_" in col:
        train_data["total logins"] += train_data[col]
        test_data["total logins"] += test_data[col]
    
    elif "s_" in col: 
        train_data["total logins with transfer"] += train_data[col]
        test_data["total logins with transfer"] += test_data[col]
    
    elif "t_" in col: 
        train_data["total logins duration"] += train_data[col]
        test_data["total logins duration"] += test_data[col]

train_data = train_data[train_data["total logins"] != 0]
test_data = test_data[test_data["total logins"] != 0]



train = []

label1 = train_data[train_data['label'] == 1] # small business owner
label0 = train_data[train_data['label'] == 0] # Non-small business owner

print("Number of small business owners: ", len(label1))
print("Number of Non-small business owners: ", len(label0))

# loc = 0
# for i in range(14):
#     if i != 13:
#         label0_ = label0[loc:loc + label1.shape[0]]
#         loc += label1.shape[0]
#         train.append(pd.concat([label1, label0_]))
#     else:
#         label0_ = label0[loc:]
#         loc += label1.shape[0]
#         train.append(pd.concat([label1, label0_]))

n = 5
train = label0.sample(47085 * n)

for i in range(1):
    train = pd.concat([train, label1])

del label1
del label0
del train_data


# Random Forest
def random_forest(train_data, test_data):
    # Split the data
    X_train = train_data.drop(['label'], axis=1)
    y_train = train_data['label']
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    rf = RandomForestClassifier()

    # Create the random grid
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X_train, y_train)

    # Predict the test data
    y_pred = rf_random.predict(X_test)

    # Print the results
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report: ", classification_report(y_test, y_pred))

    return rf_random


def XGboost(train_data, test_data):
    # Split the data
    X_train = train_data.drop(['label'], axis=1)
    y_train = train_data['label']
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    xgb = XGBClassifier(tree_method='gpu_hist')

    # Create the random grid
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    learning_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    min_child_weight = [1, 2, 3, 4]
    gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
    subsample = [0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree = [0.6, 0.7, 0.8, 0.9, 1.0]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'learning_rate': learning_rate,
                   'min_child_weight': min_child_weight,
                   'gamma': gamma,
                   'subsample': subsample,
                   'colsample_bytree': colsample_bytree}

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)

    # Fit the random search model
    xgb_random.fit(X_train, y_train)

    # Predict the test data
    y_pred = xgb_random.predict(X_test)

    # Print the results
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report: ", classification_report(y_test, y_pred))

    return xgb_random




# Neural Network
def NeuralNet(train_data, test_data, out_file):
    # Split the data
    train_data = shuffle(train_data)
    X_train = train_data.drop(['label'], axis=1)
    y_train = train_data['label']
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    # Compile model
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 1000

    # optimizer = tfa.optimizers.AdamW(
    #     learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    # )

    optimizer = Adam(learning_rate=LEARNING_RATE)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="AUC"), tf.keras.metrics.BinaryAccuracy(name="Accuracy")],
    )
    checkpoint = ModelCheckpoint(
        out_file, monitor="val_AUC", verbose=1, save_best_only=True, mode="max"
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_AUC", mode="max", patience=20, restore_best_weights=True)
    callback_list = [checkpoint, early]

    # Fit the model
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32, callbacks=callback_list, validation_data=(X_test, y_test))

    # Predict the test data
    y_pred = model.predict(X_test)

    print(f"PR AUC: {average_precision_score(y_test, y_pred.ravel())}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred.ravel())}")
    return model


def CNN_1D(train_data, test_data, out_file):
    # Split the data
    train_data = shuffle(train_data)
    X_train = train_data.drop(['label'], axis=1)
    y_train = train_data['label']
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='selu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2, activation='gelu'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=2, activation='gelu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))


    # Compile model
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 1000

    # optimizer = tfa.optimizers.AdamW(
    #     learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    # )

    optimizer = Adam(learning_rate=LEARNING_RATE)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="AUC")],
    )
    checkpoint = ModelCheckpoint(
        out_file, monitor="val_AUC", verbose=1, save_best_only=True, mode="max"
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_AUC", mode="max", patience=10, restore_best_weights=True)
    callback_list = [checkpoint, early]

    # Fit the model
    print(model.summary())
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=16, callbacks=callback_list, validation_data=(X_test, y_test))

    # Predict the test data
    y_pred = model.predict(X_test).flatten()

    print(f"PR AUC: {average_precision_score(y_test, y_pred.ravel())}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred.ravel())}")
    return model



if __name__ == "__main__":
    # for data in train:
    NeuralNetCls = CNN_1D(train, test_data, f"./models_!D_CNN/CNN_5_1_with_extra_features/")
    