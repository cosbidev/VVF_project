from src.utils_train import *
from src.utils_xai import *
import pandas as pd


def simple_pipeline(train_data: pd.DataFrame, train_labels: pd.Series, test_data: pd.DataFrame, test_labels: pd.Series, classifier: str, seed: int):
    """
    This function performs a simple pipeline used for the multiclass and binary approaches.

    Parameters
    ----------
    train_data: data to be used in the training process
    train_labels: true labels of the training data
    test_data: data to be used in the test process
    test_labels: true labels of the test data
    classifier: name of the desired model
    seed: int number used for repeatability

    Returns
    -------
    outputs: pd.DataFrame containing predictions and probabilities for eace test sample
    contributions: pd.DataFrame containing features contributions for eace test sample
    """
    predictions, probabilities, model = train_n_test(train_data, train_labels, test_data, classifier, seed)

    outputs = pd.DataFrame(dict(index=test_data.index.tolist(), label=test_labels, prediction=predictions, probability=probabilities) )
    contributions = shap_explaination(test_data, test_labels, predictions, classifier, model, test_data.columns.to_list())

    return outputs, contributions.assign(model=classifier)


def cascade_pipeline(train_data: pd.DataFrame, train_labels: pd.Series, test_data: pd.DataFrame, test_labels: pd.Series, classifier: str, seed: int):
    """
    This function performs the cascade pipeline.

    Parameters
    ----------
    train_data: data to be used in the training process
    train_labels: true labels of the training data
    test_data: data to be used in the test process
    test_labels: true labels of the test data
    classifier: name of the desired model
    seed: int number used for repeatability

    Returns
    -------
    outputs: pd.DataFrame containing predictions and probabilities for eace test sample
    contributions: pd.DataFrame containing features contributions for eace test sample
    """
    train_labels_A = train_labels.map({0: 0, 1: 0, 2: 1})
    test_labels_A = test_labels.map({0: 0, 1: 0, 2: 1})
    predictions_A, probabilities_A, model_A = train_n_test(train_data, train_labels_A, test_data, classifier, seed)

    contributions_A = shap_explaination(test_data, test_labels_A, predictions_A, classifier, model_A, test_data.columns.to_list())

    train_map_B = train_labels > 0
    train_data_B, train_labels_B = train_data.loc[train_map_B], train_labels_A.loc[train_map_B]

    test_map_B = predictions_A == 1
    if test_map_B.any():
        test_data_B, test_labels_B = test_data.loc[test_map_B], test_labels_A.loc[test_map_B]
        test_map_B = pd.Series( test_map_B, index=test_data.index)

        predictions_B, probabilities_B, model_B = train_n_test(train_data_B, train_labels_B, test_data_B, classifier, seed)
        contributions_B = shap_explaination(test_data_B, test_labels_B, predictions_B, classifier, model_B, test_data_B.columns.to_list())

        predictions_B = pd.Series(predictions_B + 1, index=test_data_B.index.tolist())
    else:
        predictions_B = pd.DataFrame()
        probabilities_B = pd.DataFrame()
        contributions_B = pd.DataFrame()

    outputs = pd.DataFrame(dict(index=test_data.index.tolist(), label=test_labels, prediction=predictions_A, probability=probabilities_A.values.tolist()) )

    for i, possible_relevant in test_map_B.items():
        if not possible_relevant:
            not_relevant_probability = probabilities_A[i][0]
            interesting_probility = probabilities_A[i][1]
            relevant_probility = probabilities_A[i][1]
        else:
            not_relevant_probability = probabilities_A[i][0]
            interesting_probility = (probabilities_A[i][1] + probabilities_B[i][0])/2
            relevant_probility = (probabilities_A[i][1] + probabilities_B[i][1])/2

            outputs.loc[i, "prediction"] = predictions_B[i]

        outputs.at[i, "probability"] = np.array([not_relevant_probability, interesting_probility, relevant_probility])

    contributions = pd.concat([contributions_A.assign(model=f"{classifier}_A"), contributions_B.assign(model=f"{classifier}_B") ], axis=0, ignore_index=True )

    return outputs, contributions


if __name__ == "__main__":
    pass
