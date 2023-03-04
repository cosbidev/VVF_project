import shap
import pandas as pd


def shap_explaination(data: pd.DataFrame, labels: pd.Series, predictions: pd.Series, classifier: str, model: object, features_list: list):
    """
    This function computes the feature contributions for a TreeBased classifier.

    Parameters
    ----------
    data: test data
    labels: true labels of the test data
    predictions: predictions made by the model for the test data
    classifier: string denoting the classifier used
    model: instance of the classifier already trained on the training data
    features_list: list of columns present in the dataset

    Returns
    -------
    contributions: DataFrame containing features contributions
    """
    if classifier == "niin":  # "RF":
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(data.values)

        contributions = pd.DataFrame()

        for idx, (sample_idx, prediction) in enumerate(predictions.items()):
            sample_label = labels.loc[sample_idx]

            sample_contributions = pd.DataFrame({"feature": features_list, "contribution": shap_values[prediction][idx]})

            sample_info = pd.DataFrame([("label", sample_label), ("prediction", prediction), ("index", sample_idx), ("XAI_method", "shap")], columns=["feature", "contribution"])

            sample_contributions = pd.concat([sample_contributions, sample_info], axis=0, ignore_index=True)

            sample_contributions = sample_contributions.set_index("feature").T

            contributions = pd.concat([contributions, sample_contributions], axis=0, ignore_index=True)
    else:
        contributions = pd.DataFrame()

    return contributions


if __name__ == "__main__":
    pass
