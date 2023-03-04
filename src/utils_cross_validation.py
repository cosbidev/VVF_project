import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut


def set_cross_validation(IDs: pd.Series, labels: pd.Series, directory: str, cross_validation_method: str, seed: int):
    """
    This function creates the folds to be used in the cross validation and saves them in an Excel file.

    Parameters
    ----------
    IDs: list of IDs of the samples
    labels: list of the labels of the samples
    directory: path to the directory where to save the cross validation file
    cross_validation_method: method to be used to create the folds, stratifiedkfold or loo
    seed: seed number to be used in the creation of the folds

    Returns
    -------
    path: path to the file where the cv-folds are being saved
    """
    path = os.path.join(directory, "cross_validation", f"{cross_validation_method}.csv")

    cv_options = {"stratifiedkfold": StratifiedKFold, "loo": LeaveOneOut}
    cvparams_options = {"stratifiedkfold": dict(n_splits=10, shuffle=True, random_state=seed), "loo": {}}

    cv = cv_options[cross_validation_method](**cvparams_options[cross_validation_method])

    folds = pd.DataFrame([test_index for _, test_index in cv.split(IDs, labels)])

    samples = pd.DataFrame(dict(ID=IDs, label=labels))
    for fold_number, fold_idxs in folds.iterrows():
        samples.loc[fold_idxs.dropna(), "fold"] = fold_number

    samples.to_csv(path)
    return path


def get_cross_validation(path: str):
    """
    This function returns the folds computed in the "set_cross_validation" function.

    Parameters
    ----------
    path: path to the file containing the folds

    Yields
    -------
    train_folds, test_fold: info about the train and test folds
    """
    samples = pd.read_csv(path, index_col=0)

    for fold_number in sorted(samples.fold.unique()):
        train_map = samples.fold != fold_number
        test_map = samples.fold == fold_number
        yield samples.loc[train_map], samples.loc[test_map]


if __name__ == "__main__":
    pass
