import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils import shuffle
# transformer
from tabtransformertf.models.tabtransformer import TabTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep
from tabtransformertf.utils.preprocessing import build_numerical_prep
from sklearn.metrics import average_precision_score, roc_auc_score

# path = "/content/drive/MyDrive/Colab Notebooks/College/data-science comp/"
path = ""

# Read the data
train_data = pd.read_pickle(path + "train_task1.pkl")
test_data = pd.read_pickle(path + "test_task1.pkl")

train = []

label1 = train_data[train_data['label'] == 1]  # small business owner
label0 = train_data[train_data['label'] == 0]  # Non-small business owner

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

catagorical = ['gender_1', 'gender_2', 'age_code_1', 'age_code_2', 'age_code_3',
               'age_code_4', 'age_code_5', 'age_code_6', 'age_code_7', 'age_code_8',
               'age_code_9', 'age_code_10', 'age_code_11', 'age_code_12',
               'age_code_13', 'age_code_14', 'region_code_0', 'region_code_1',
               'region_code_2', 'region_code_4', 'region_code_5', 'region_code_6',
               'region_code_7', 'region_code_8', 'region_code_9', 'region_code_10',
               'region_code_11', 'region_code_12', 'region_code_13', 'region_code_14',
               'region_code_15', 'region_code_16', 'region_code_17', 'region_code_18']

numerrical = ['c_week1',
              'c_week2',
              'c_week3',
              'c_week4',
              'c_week5',
              'c_week6',
              'c_week7',
              'c_week8',
              'c_week9',
              'c_week10',
              'c_week11',
              'c_week12',
              'c_week13',
              'c_week14',
              'c_week15',
              'c_week16',
              'c_week17',
              'c_week18',
              'c_week19',
              'c_week20',
              'c_week21',
              'c_week22',
              'c_week23',
              'c_week24',
              'c_week25',
              'c_week26',
              'c_week27',
              'c_week28',
              'c_week29',
              'c_week30',
              'c_week31',
              'c_week32',
              'c_week33',
              'c_week34',
              's_week1',
              's_week2',
              's_week3',
              's_week4',
              's_week5',
              's_week6',
              's_week7',
              's_week8',
              's_week9',
              's_week10',
              's_week11',
              's_week12',
              's_week13',
              's_week14',
              's_week15',
              's_week16',
              's_week17',
              's_week18',
              's_week19',
              's_week20',
              's_week21',
              's_week22',
              's_week23',
              's_week24',
              's_week25',
              's_week26',
              's_week27',
              's_week28',
              's_week29',
              's_week30',
              's_week31',
              's_week32',
              's_week33',
              's_week34',
              't_week1',
              't_week2',
              't_week3',
              't_week4',
              't_week5',
              't_week6',
              't_week7',
              't_week8',
              't_week9',
              't_week10',
              't_week11',
              't_week12',
              't_week13',
              't_week14',
              't_week15',
              't_week16',
              't_week17',
              't_week18',
              't_week19',
              't_week20',
              't_week21',
              't_week22',
              't_week23',
              't_week24',
              't_week25',
              't_week26',
              't_week27',
              't_week28',
              't_week29',
              't_week30',
              't_week31',
              't_week32',
              't_week33',
              't_week34']


def train_model(train_data, test_data, catagorical, numerrical, out_file):
    train_data[catagorical] = train_data[catagorical].astype(str)
    test_data[catagorical] = test_data[catagorical].astype(str)

    train_data[numerrical] = train_data[numerrical].astype(float)
    test_data[numerrical] = test_data[numerrical].astype(float)

    FEATURES = catagorical + numerrical
    LABEL = "label"

    # Transform to TF dataset
    train_dataset = df_to_dataset(train_data[FEATURES + [LABEL]], LABEL, batch_size=1024)
    val_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], LABEL, shuffle=False, batch_size=1024)
    category_prep_layers = build_categorical_prep(train_data, catagorical)

    # Create model
    tabtransformer = TabTransformer(
        numerical_features=numerrical,  # List with names of numeric features
        categorical_features=catagorical,  # List with names of categorical feature
        categorical_lookup=category_prep_layers,  # Dict with StringLookup layers
        numerical_discretisers=None,  # None, we are simply passing the numeric features
        embedding_dim=32,  # Dimensionality of embeddings
        out_dim=1,  # Dimensionality of output (binary task)
        out_activation='sigmoid',  # Activation of output layer
        depth=4,  # Number of Transformer Block layers
        heads=8,  # Number of attention heads in the Transformer Blocks
        attn_dropout=0.1,  # Dropout rate in Transformer Blocks
        ff_dropout=0.1,  # Dropout rate in the final MLP
        mlp_hidden_factors=[2, 4],  # Factors by which we divide final embeddings for each layer
        use_column_embedding=True,  # If we want to use column embeddings
    )

    # Compile model
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 1000

    optimizer = tfa.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    tabtransformer.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="PR AUC", curve='PR')],
    )

    checkpoint = ModelCheckpoint(
        out_file, monitor="val_PR AUC", verbose=1, save_best_only=True, mode="max"
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_PR AUC", mode="max", patience=10, restore_best_weights=True)
    callback_list = [checkpoint, early]

    tf.keras.backend.clear_session()
    tabtransformer.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=val_dataset,
        callbacks=callback_list,
    )
    val_preds = tabtransformer.predict(val_dataset)

    print(f"PR AUC: {average_precision_score(test_data['label'], val_preds.ravel())}")
    print(f"ROC AUC: {roc_auc_score(test_data['label'], val_preds.ravel())}")

    return tabtransformer


def main():
    model_num = 0
    for data in train:
        model_num += 1
        scaler = StandardScaler()
        data[numerrical] = scaler.fit_transform(data[numerrical])
        test_data[numerrical] = scaler.transform(test_data[numerrical])
        train_model(data, test_data, catagorical, numerrical, f"model_{model_num}")


main()