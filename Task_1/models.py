import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from xgboost import XGBClassifier

path = "/content/drive/MyDrive/Colab Notebooks/College/data-science comp/"

# Read the data
train_data = pd.read_pickle(path + "train_task1.pkl")
test_data = pd.read_pickle(path + "test_task1.pkl")

train = []

label1 = train_data[train_data['label'] == 1] # small business owner
label0 = train_data[train_data['label'] == 0] # Non-small business owner

print("Number of small business owners: ", len(label1))
print("Number of Non-small business owners: ", len(label0))

loc = 0
for i in range(14):
    if i != 13:
        label0_ = label0[loc:loc + label1.shape[0]]
        loc += label1.shape[0]
        train.append(pd.concat([label1, label0_]))
    else:
        label0_ = label0[loc:]
        loc += label1.shape[0]
        train.append(pd.concat([label1, label0_]))


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


