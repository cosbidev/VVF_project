import numpy as np
import pandas as pd
from src.utils_features import *
from src.utils_cross_validation import *
from src.utils_preprocessing import *
from src.utils_pipeline import *
from src.utils_results import *


def main():
    data_path = "./data"
    output_path = "./outputs"

    cfg = dict(
        seed = 100,
        cross_val = "stratifiedkfold",
        classification = "cascade",  # "binary", "multiclass", "cascade"
        classifier = "RF",  # "NN", "RF", "SVM"
        max_upd = 6,
        ablation = False
    )

    classes = {True: ["non_rilevante", "rilevante"], False: ["non_rilevante", "interessante", "rilevante"]}
    maps = {True: dict(non_rilevante=0, interessante=0, rilevante=1), False: dict(non_rilevante=0, interessante=1, rilevante=2)}
    classes_map = maps[ cfg["classification"] == "binary" ]
    classes = classes[ cfg["classification"] == "binary" ]
    del maps

    np.random.seed = cfg["seed"]
    main_data_path = os.path.join( data_path, "main_data")
    db = load_dataset("dataset.xlsx", main_data_path )

    print(db.label.value_counts())

    additional_data_path = os.path.join( data_path, "additional_data")
    db_test = load_dataset("dataset_test.xlsx", additional_data_path)

    print(db_test.label.value_counts())

    # features_computation(db, main_data_path, max_upd=cfg["max_upd"])
    # features_computation(db_test, additional_data_path, max_upd=cfg["max_upd"])

    vehicles_list = np.unique([ *db.vehicle_type.explode().unique().tolist(), *db_test.vehicle_type.explode().unique().tolist() ]).tolist()

    print( f"{len(vehicles_list)} possible vehicles")
    descriptions_list = np.unique([ *db.description.unique().tolist(), *db_test.description.unique().tolist() ]).tolist()
    print( f"{len(descriptions_list)} possible descriptions")

    cv_path = set_cross_validation(db.intervention_number.values, db.label.values, main_data_path, cfg["cross_val"], cfg["seed"])

    total_outputs = list()
    total_test_outputs = list()
    total_performance = pd.DataFrame()
    total_test_performance = pd.DataFrame()
    total_contributions = pd.DataFrame()
    total_test_contributions = pd.DataFrame()

    for upd in range(1, cfg["max_upd"] + 1):
        print(f"Update {upd}")
        update_features = load_update_features(main_data_path, upd)
        additional_test_data = load_update_features(additional_data_path, upd)

        update_features = categorical_preprocessing(update_features, dichotomise=True)
        additional_test_data = categorical_preprocessing(additional_test_data, dichotomise=True, columns = update_features.columns)

        update_features = emergency_vehicles_preprocessing(update_features, vehicles_list)
        additional_test_data = emergency_vehicles_preprocessing(additional_test_data, vehicles_list)

        update_outputs = pd.DataFrame()
        update_contributions = pd.DataFrame()
        for fold, (train_index, test_index) in enumerate(get_cross_validation(cv_path)):
            print( f"Fold {fold}")
            train_labels, test_labels = update_features.loc[train_index.index, "label"], update_features.loc[test_index.index, "label"]
            update_features_wo_label = update_features.drop("label", axis=1)

            train_labels, test_labels = train_labels.map(classes_map), test_labels.map(classes_map)

            train_probabilities, test_probabilities = get_prior_probabilities(train_labels, classes_map, classes, train_index, test_index)
            if upd > 1 and not cfg["ablation"]:
                for i in range(upd-1):
                    previous_probabilities = total_outputs[i]["probability"].apply(pd.Series).rename({idx: f"probability_{cl}" for idx, cl in enumerate(classes)}, axis=1)
                    train_probabilities, test_probabilities = train_probabilities + previous_probabilities.loc[train_index.index], test_probabilities + previous_probabilities.loc[test_index.index]

            train_data, test_data = update_features_wo_label.iloc[train_index.index], update_features_wo_label.iloc[test_index.index]

            if not cfg["ablation"]:
                train, test = pd.concat([train_data, train_probabilities], axis=1), pd.concat([test_data, test_probabilities], axis=1)
            else:
                train, test = train_data.copy(), test_data.copy()

            train_scaled, test_scaled = features_standardization( train, test )

            if cfg["classification"] != "cascade":
                outputs, contributions = simple_pipeline(train_scaled, train_labels, test_scaled, test_labels, cfg["classifier"], cfg["seed"])
            else:
                outputs, contributions = cascade_pipeline(train_scaled, train_labels, test_scaled, test_labels, cfg["classifier"], cfg["seed"])

            update_outputs = pd.concat( [update_outputs, outputs], axis=0)
            update_contributions = pd.concat( [update_contributions, contributions.assign(fold=fold, update=upd)], axis=0 )

        update_performance = evaluate_performance(update_outputs.label.values, update_outputs.prediction.values, classes)
        update_performance = update_performance.assign(update=upd).reset_index()

        total_performance = pd.concat( [total_performance, update_performance], axis=0, ignore_index=True)

        total_outputs.append(update_outputs)
        total_contributions = pd.concat( [total_contributions, update_contributions], axis=0)

        train_labels, additional_test_labels = update_features.label, additional_test_data.label
        update_features_wo_label, additional_test_data_wo_label = update_features.drop("label", axis=1), additional_test_data.drop("label", axis=1)

        train_labels, additional_test_labels = train_labels.map(classes_map), additional_test_labels.map(classes_map)
        train_probabilities, additional_test_probabilities = get_prior_probabilities(train_labels, classes_map, classes, update_features_wo_label, additional_test_data_wo_label)
        if upd > 1:
            for i in range(upd-1):
                train_probabilities = train_probabilities + total_outputs[i]["probability"].apply(pd.Series).rename({idx: f"probability_{cl}" for idx, cl in enumerate(classes)}, axis=1)
                additional_test_probabilities = additional_test_probabilities + total_test_outputs[i]["probability"].apply(pd.Series).rename({idx: f"probability_{cl}" for idx, cl in enumerate(classes)}, axis=1)

        train, additional_test = pd.concat([update_features_wo_label, train_probabilities], axis=1), pd.concat([additional_test_data_wo_label, additional_test_probabilities], axis=1)

        train_scaled, additional_test_scaled = features_standardization( train, additional_test )

        if cfg["classification"] != "cascade":
            additional_outputs, additional_contributions = simple_pipeline(train_scaled, train_labels, additional_test_scaled, additional_test_labels, cfg["classifier"], cfg["seed"])
        else:
            additional_outputs, additional_contributions = cascade_pipeline(train_scaled, train_labels, additional_test_scaled, additional_test_labels, cfg["classifier"], cfg["seed"])

        upd_test_performance = evaluate_performance(additional_outputs.label.values, additional_outputs.prediction.values, classes)
        upd_test_performance = upd_test_performance.assign(update=upd).reset_index()

        total_test_outputs.append( additional_outputs )
        total_test_performance = pd.concat( [total_test_performance, upd_test_performance], axis=0, ignore_index=True)
        total_test_contributions = pd.concat( [total_test_contributions, additional_contributions.assign(update=upd) ], axis=0)

    total_performance = total_performance
    total_test_performance = total_test_performance

    total_performance.to_excel(os.path.join( output_path, "total_performance.xlsx"), index=False)
    total_test_performance.to_excel(os.path.join( output_path, "additional_total_performance.xlsx"), index=False)

    folds_performance = total_performance.groupby(by=["class", "update"]).agg("mean")
    folds_performance.to_excel(os.path.join( output_path, "updates_performance.xlsx"))

    total_contributions.to_csv( os.path.join( output_path, "features_contributions.csv"), index=False)
    total_test_contributions.to_csv( os.path.join( output_path, "additional_features_contributions.csv"), index=False)


if __name__ == '__main__':
    main()
