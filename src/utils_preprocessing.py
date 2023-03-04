import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def categorical_preprocessing(data: pd.DataFrame, dichotomise: bool = True, columns: list = None):
    """
    This function performs the categorical preprocessing (i.e., one-hot-encoding or categorical enconding).

    Parameters
    ----------
    data: data to preprocess
    dichotomise: boolean value indicating whether to dichotomise or not the data
    columns: if given, columns present in the training data in order to adapt the test data

    Returns
    -------
    categorical_data: preprocessed dataframe
    """
    categorical_data = data.copy()

    categorical_data.elevation_zone = categorical_data.elevation_zone.astype('category')
    categorical_data.mountain_municipality = categorical_data.mountain_municipality.astype('category')
    categorical_data.urbanisation_degree = categorical_data.urbanisation_degree.astype('category')
    categorical_data.description = categorical_data.description.astype('category')
    categorical_data.coastal_municipality = categorical_data.coastal_municipality.astype('category')

    if dichotomise:
        categorical_data = pd.get_dummies(categorical_data, columns=["elevation_zone", "mountain_municipality", "urbanisation_degree", "description"]).fillna(0)
        categorical_data.coastal_municipality = categorical_data.coastal_municipality.astype(np.int8)
    else:
        categorical_data.elevation_zone = categorical_data.elevation_zone.cat.codes
        categorical_data.mountain_municipality = categorical_data.mountain_municipality.cat.codes
        categorical_data.urbanisation_degree = categorical_data.urbanisation_degree.cat.codes
        categorical_data.description = categorical_data.description.cat.codes
        categorical_data.coastal_municipality = categorical_data.coastal_municipality.cat.codes

    if columns is not None:
        categorical_data = pd.concat( [pd.DataFrame(columns=columns), categorical_data], axis=0).fillna(0)

    return categorical_data


def emergency_vehicles_preprocessing(data: pd.DataFrame, vehicles_list: list):
    """
    This function performs the preprocessing for the emergency means.

    Parameters
    ----------
    data: data to be processed
    vehicles_list: list of vehicles present in the whole dataset

    Returns
    -------
    data_processed
    """
    vehicles_features = pd.DataFrame(columns=vehicles_list)
    vehicles_dummies = pd.get_dummies(data.emergency_means.apply(eval).explode()).reset_index().groupby(by="index").sum()
    vehicles_features = pd.concat([vehicles_features, vehicles_dummies], axis=0).fillna(0)

    data_processed = pd.concat([data.drop('emergency_means', axis=1), vehicles_features], axis=1)

    return data_processed


def standardize(data: pd.DataFrame, scaler: object = None):
    """
    This function standardize the numerical features.

    Parameters
    ----------
    data: data to be processed
    scaler: scaler to use to standardize the input data

    Returns
    -------
    data_scaled: input data with scaled numerical featurees
    scaler: StandardScaler object
    """
    features_to_standardize = ["centre_altitude", "hour_of_the_day", "month", "update_number", "first_departure", "event_duration", "average_update_time"]

    data_to_scale = data.copy()

    if not scaler:
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data_to_scale.loc[:, features_to_standardize].values), columns=features_to_standardize, index=data_to_scale.index)
    else:
        data_scaled = pd.DataFrame(scaler.transform(data_to_scale.loc[:, features_to_standardize].values), columns=features_to_standardize, index=data_to_scale.index)

    data_scaled = pd.concat([data_scaled, data_to_scale.drop(features_to_standardize, axis=1)], axis=1)

    return data_scaled, scaler


def features_standardization(data: pd.DataFrame, test_data: pd.DataFrame):
    """

    Parameters
    ----------
    data: train data to fit the StandardScaler object
    test_data: test data to be transformed by the StandardScaler

    Returns
    -------
    scaled_data
    scaled_test_data
    """
    scaled_data, scaler = standardize(data)

    scaled_test_data, _ = standardize(test_data, scaler)

    return scaled_data, scaled_test_data


if __name__ == "__main__":
    pass
