from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd


def evaluate_performance(labels: np.ndarray, predictions: np.ndarray, classes: list):
    """
    This function computes the performance.

    Parameters
    ----------
    labels: list of true labels
    predictions: list of predictions made by the model
    classes: list of classes present in the dataset

    Returns
    -------
    performance: pd.DataFrame containing the performance for each class

    """
    metrics = dict(accuracy=accuracy_score, recall=recall_score, precision=precision_score, f1_score=f1_score)

    performance = dict()
    for metric_name, metric in metrics.items():
        if metric_name == "accuracy":
            perf = metric(y_true=labels, y_pred=predictions)
        else:
            perf = metric(y_true=labels, y_pred=predictions, average=None, zero_division=0, labels=np.arange(len(classes)))

        performance[metric_name] = np.round(perf, 4)*100

    performance = pd.DataFrame(performance, index=classes).rename_axis("class", axis=0)
    return performance


if __name__ == "__main__":
    pass
