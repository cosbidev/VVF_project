import os
import numpy as np
import pandas as pd
from src.utils_data import *


def average_update_time(event: pd.Series, upd: int):
    """
    This function computes the average update time feature.

    Parameters
    ----------
    event: pandas Series containing the event information
    upd: number of the update under consideration

    Returns
    -------
    average_update_time: float - mean of the time elapsed between different updates of a single event
    """
    if upd < event.updates_number:
        updates_times = [event.call_time] + list(event.assignment_time[0:upd])
    else:
        updates_times = [event.call_time] + list(event.assignment_time) + [event.closing_time]

    return pd.Series(updates_times).diff().dropna().mean().total_seconds() / 60


def event_duration(event: pd.Series, upd: int):
    """
    This function computes the duration of the event.

    Parameters
    ----------
    event: pandas Series containing the event information
    upd: number of the update under consideration

    Returns
    -------
    event_duration: float - time elapsed since the time of the call
    """
    if upd < event.updates_number:
        end_time = event.assignment_time[upd]
    else:
        end_time = event.closing_time

    return (end_time - event.call_time).total_seconds() / 60


def static_features_computation(db: pd.DataFrame, path: str):
    """
    This function computes the static features.

    Parameters
    ----------
    db: dataset to be used to compute the features
    path: path where to load ISTAT and reduced description files and to save the features computed

    Returns
    -------
    None
    """
    reduced_descriptions = load_reduced_description(path)
    ISTAT_data = load_ISTAT_data(path)

    # ISTAT features
    dataset = db.municipality.apply(lambda name: ISTAT_data.loc[name])

    # hour of the day [int]
    dataset = dataset.assign(hour_of_the_day=db.call_time.apply(lambda date: date.hour).astype("int"))

    # month [int]
    dataset = dataset.assign(month=db.call_time.apply(lambda date: date.month).astype("int"))

    # time first departure [min]
    dataset = dataset.assign(first_departure=db.apply(lambda event: (event.first_out_time - event.call_time).total_seconds() / 60, axis=1).astype("float")).fillna(0)

    # descrizione [category]
    dataset = dataset.assign(description=db.description.apply(lambda description: reduced_descriptions.loc[description, "reduced_description"]).astype('category'))

    dataset = dataset.assign(label=db.label.astype('category'))

    filename_path = os.path.join(path, "features", "static_features.csv")
    dataset.to_csv(filename_path, index=False)


def dynamic_features_computation(db: pd.DataFrame, path: str, upd: int):
    """
    This function computes the dynamic features.

    Parameters
    ----------
    db: dataset to be used to compute the features
    path: path where to load ISTAT and reduced description files and to save the features computed
    upd: number of the update under consideration

    Returns
    -------
    None
    """
    # numero di aggiornamenti [int]
    dataset = pd.DataFrame({'update_number': db.updates_number.apply(lambda n_upd: min(n_upd, upd)).astype("int")})

    # lista mezzi
    dataset = dataset.assign(emergency_means=db.vehicle_type.apply(lambda means_list: means_list[0:upd]))

    # tempo medio di aggiornamento [minuti]
    dataset = dataset.assign(average_update_time=db.apply(average_update_time, upd=upd, axis=1))

    # durata evento [minuti]
    dataset = dataset.assign(event_duration=db.apply(event_duration, upd=upd, axis=1))  # .fillna(-1))

    filename_path = os.path.join(path, "features", "dynamic_features_{}.csv".format(upd))
    dataset.to_csv(filename_path, index=False)


def features_computation(db: pd.DataFrame, path: str, max_upd: int):
    """
    This function computes the features.

    Parameters
    ----------
    db: data to use to compute the features
    path: path where to save the features
    max_upd: max update to compute the features

    Returns
    -------
    None
    """
    static_features_computation(db, path)

    for upd in np.arange(1, max_upd + 1):
        print(f"Computing features for update #{upd}")
        dynamic_features_computation(db, path, upd)


def get_prior_probabilities(train_labels: pd.Series, classes_map: dict, classes: list, train_index: pd.DataFrame, test_index: pd.DataFrame):
    """
    This function computes the prior probabilities from the training samples and assigns them to all the samples.

    Parameters
    ----------
    train_labels: true labels of the train samples
    classes_map: dict to map the labels to their relative number (i.e., 0: not relevant, 1: interesting, 2: relevant)
    classes: list of the classes in the dataset
    train_index: DataFrame containing the train data indexes
    test_index: DataFrame containing the test data indexes

    Returns
    -------
    train_probabilities: pandas DataFrame containing the prior probabilities for the training data
    test_probabilities: pandas DataFrame containing the prior probabilities for the test data
    """
    prior_probabilities = train_labels.value_counts(normalize=True).sort_index()

    train_probabilities = pd.DataFrame(index=train_index.index, columns=[f"probability_{cl}" for cl in classes_map.keys()])
    test_probabilities = pd.DataFrame(index=test_index.index, columns=[f"probability_{cl}" for cl in classes_map.keys()])
    for idx, cl in enumerate(classes):
        train_probabilities[f"probability_{cl}"] = prior_probabilities[idx]
        test_probabilities[f"probability_{cl}"] = prior_probabilities[idx]

    return train_probabilities, test_probabilities


if __name__ == "__main__":
    pass
