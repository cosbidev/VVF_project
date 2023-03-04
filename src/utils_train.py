import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


def train_model(data: np.ndarray, labels: np.ndarray, classifier: str, seed: int = 42):
    """
    This function instantiate the model and trains it on the training data.

    Parameters
    ----------
    data: data to be used in the training process
    labels: labels to be used in the training process
    classifier: name of the classifier to be used
    seed: seed number used for repeatability

    Returns
    -------
    model: the trained model
    """
    verbose_mode = 0

    if classifier == "NN":
        cat_labels = to_categorical(labels)

        n_input = data.shape[1]
        n_output = cat_labels.shape[1]

        K.clear_session()

        model = Sequential()
        model.add(Dense(150, input_shape=(n_input,), activation='sigmoid', kernel_initializer='he_normal'))
        model.add(Dense(100, activation='sigmoid', kernel_initializer='he_normal'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(30, activation='sigmoid', kernel_initializer='he_normal'))
        model.add(Dense(n_output, activation='softmax', kernel_initializer='he_normal'))

        model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=verbose_mode)

        callbacks_list = [early_stop]

        model.fit(data, cat_labels, epochs=100, batch_size=32, validation_split=0.1, verbose=verbose_mode, callbacks=callbacks_list)

    elif classifier == "RF":

        model = RandomForestClassifier(criterion='entropy',
                                       n_estimators=100,
                                       max_features='sqrt',
                                       bootstrap=False,
                                       verbose=verbose_mode,
                                       random_state=seed,
                                       n_jobs=-1, class_weight="balanced")

        model.fit(data, labels)

    else:

        model = SVC(decision_function_shape='ovo',
                    kernel='rbf',
                    random_state=seed,
                    verbose=verbose_mode,
                    probability=True)

        model.fit(data, labels)

    return model


def test_model(data: np.ndarray, model: object, classifier: str):
    """
    This function tests the model on the test data.

    Parameters
    ----------
    data: data to be used in the test process
    model: trained model to be tested
    classifier: name of the classifier

    Returns
    -------
    predictions: np.ndarray containing the predictions made by the model
    probabilities: np.ndarray containing the probabilities returned by the model

    """
    if classifier == "NN":
        probabilities = model.predict(data)
        predictions = np.argmax(probabilities, axis=1)

    else:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)

    return predictions, probabilities


def train_n_test(data: pd.DataFrame, labels: pd.Series, test_data: pd.DataFrame, classifier: str, seed: int):
    """
    This function trains the desired model on the training data and then it tests it in the test_data.

    Parameters
    ----------
    data: data to be used in the training process
    labels: true labels of the training data
    test_data: data to be used in the test process
    classifier: name of the desired classifier
    seed: int number to be used for repeatability

    Returns
    -------
    predictions: np.ndarray containing the predictions made by the model on the test data
    probabilities: np.ndarray containing the probabilities returned by the model on the test data
    model: trained model

    """
    model = train_model(data.values, labels.values, classifier, seed)

    predictions, probabilities = test_model(test_data.values, model, classifier)

    predictions = pd.Series(predictions, index=test_data.index)
    probabilities = pd.Series(probabilities.tolist(), index=test_data.index)

    return predictions, probabilities, model


if __name__ == "__main__":
    pass
