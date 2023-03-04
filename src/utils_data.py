import os
import pandas as pd


def load_dataset(filename: str, path: str):
    """
    This function loads the dataset.

    Parameters
    ----------
    filename: name of the file
    path: path to the folder containing the file to load

    Returns
    -------
    db: dataset loaded
    """
    db = pd.read_excel(os.path.join(path, "dataset", filename))

    columns_w_lists = ["vehicle_type", "team_abbreviation", "team_shift", "sms_id", "sms_scores"]
    columns_w_dates_lists = ["assignment_time", "departure_time", "return_time", "service_date", "sms_time"]

    for col in columns_w_lists + columns_w_dates_lists:
        db[ col ] = db[ col ].apply(eval)

    for col in columns_w_dates_lists:
        db[ col ] = db[ col ].apply( lambda x: [pd.to_datetime(y) for y in x] )

    return db


def municipality_correction(name: str):
    """
    This function changes some of the municipality names to align the names in the dataset and in the ISTAT dataset-

    Parameters
    ----------
    name: name of the municipality

    Returns
    -------
    name: corrected name

    """
    name = " ".join( name.lower().replace("'", " ").replace("-", " ").split() )

    municipalities_dict = {"leinì": "leini",
                            "borgoforte": "borgo virgilio",  # in realtà è una frazione quindi non c'è nella lista}
                            "maccagno": "maccagno con pino e veddasca",
                            "giugliano": "giugliano in campania",
                            "mugnano": "mugnano di napoli",
                            "castellammare": "castellammare di stabia",
                            "verderio inferiore": "verderio",
                            "gravedona": "gravedona ed uniti",
                            "lenno": "tremezzina",  # ex comune
                            "cesano": "cesano boscone",  # scelgo Boscone perchè più vicino a milano, non Maderno
                            "cavenago": "cavenago di brianza",  # scelgo di Brianza perchè più vicino a milano, non d'Adda
                            "rivanazzano": "rivanazzano terme",
                            "rovagnate": "la valletta brianza"}  # sede del comune

    if name in municipalities_dict:
        name = municipalities_dict[name]

    return name


def load_ISTAT_data(path: str):
    """
    This function loads the ISTAT dataset.

    Parameters
    ----------
    path: path to the folder containing the ISTAT dataset

    Returns
    -------
    ISTAT_data: loaded ISTAT dataset
    """
    file_path = os.path.join(path, "features_info", "Classificazioni_statistiche_comuni_30_06_2017.txt")

    converters = {"Denominazione_(solo_italiano)": municipality_correction}
    dtypes = {"Zona_altimetrica": "int", "Altitudine_del_centro_(metri)": "float", "Comune_litoraneo": "int", "Comune_Montano": 'category', "Grado_di_urbanizzazione": "int"}

    ISTAT_data = pd.read_csv(file_path, sep=";", header=0, encoding='mac_roman', usecols=list(dtypes.keys()) + ["Denominazione_(solo_italiano)"], dtype=dtypes, converters=converters)

    new_names = {"Denominazione_(solo_italiano)": "municipality", "Zona_altimetrica": "elevation_zone",
                 "Altitudine_del_centro_(metri)": "centre_altitude", "Comune_litoraneo": "coastal_municipality",
                 "Comune_Montano": "mountain_municipality", "Grado_di_urbanizzazione": "urbanisation_degree"}
    ISTAT_data = ISTAT_data.rename(new_names, axis=1)

    ISTAT_data = ISTAT_data.set_index("municipality")  # "Denominazione_(solo_italiano)")

    ISTAT_data = ISTAT_data[~ISTAT_data.index.duplicated(keep='first')]
    return ISTAT_data


def load_reduced_description(path: str):
    """
    This function loads the file that contains the descriptions mapping.

    Parameters
    ----------
    path: path to the folder containing the file with the descriptions mapping.

    Returns
    -------
    descriptions: loaded data
    """
    file_path = os.path.join(path, "features_info", "descrizione_ridotta.csv")
    descriptions = pd.read_csv(file_path, sep=',', index_col=False, header=0, names=["description", "reduced_description"])
    descriptions = descriptions.set_index("description")
    return descriptions


def load_update_features(path: str, upd: int):
    """
    This function loads the features of a specific update.

    Parameters
    ----------
    path: path to the folder containing the features
    upd: number of the update under consideration

    Returns
    -------
    upd_features: features computed at the update under consideration
    """
    static_features_file_path = os.path.join( path, "features", "static_features.csv" )
    dynamic_features_file_path = os.path.join( path, "features", f"dynamic_features_{upd}.csv" )

    static_features = pd.read_csv(static_features_file_path)
    dynamic_features = pd.read_csv(dynamic_features_file_path)

    upd_features = pd.concat( [static_features, dynamic_features], axis=1)
    return upd_features


if __name__ == "__main__":
    pass
