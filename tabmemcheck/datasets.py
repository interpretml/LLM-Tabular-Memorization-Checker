import numpy as np
import pandas as pd

import importlib.resources as resources
import yaml

import tabmemcheck.utils as utils


ORIGINAL_VERSION = "original"
PERTURBED_VERSION = "perturbed"
TASK_VERSION = "task"
STATISTICAL_VERSION = "statistical"

DATASET_VERSIONS = [
    ORIGINAL_VERSION,
    PERTURBED_VERSION,
    TASK_VERSION,
    STATISTICAL_VERSION,
]


def __validate_inputs(version):
    assert version in DATASET_VERSIONS, (
        "dataset version must be one of "
        + ", ".join(DATASET_VERSIONS)
        + f"got {version}"
    )


def __load_yaml_config(dataset_name: str):
    """Load from the resources folder of the package"""
    with resources.open_text(
        "tabmemcheck.resources.config", f"{dataset_name}.yaml"
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def __check_perturbed_rows(df_original, df_perturbed):
    df_common = pd.merge(df_original, df_perturbed, how="inner")
    if df_common.empty:
        print("None of the perturbed rows appear in the original dataset.")
    else:
        per_cent = 100.0 * df_common.shape[0] / df_perturbed.shape[0]
        print(
            f"{df_common.shape[0]} perturbed rows appear in the original dataset (that is {per_cent:.2f}% of all perturbed rows)."
        )


def __apply_perturbations(df: pd.DataFrame, config: dict, seed=None):
    """Perturb the values of the features according to the perturbations specified in the configuration file."""
    df = df.copy(deep=True)
    PERTURBATION_METHODS = {
        "integer_perturbation": integer_perturbation,
        "value_perturbation": value_perturbation,
    }
    perturbations = config["Perturbations"]  # a dict with the perturbations
    for feature_name in perturbations.keys():
        if feature_name in df.columns:
            schedule = perturbations[feature_name]
            method = schedule[0]  # string
            parameters = schedule[1]  # dict with key-word arguments
            parameters["seed"] = seed  # add the seed
            assert (
                method in PERTURBATION_METHODS.keys()
            ), f"Unknown perturbation method {method}."
            pert_fn = PERTURBATION_METHODS[method]
            df[feature_name] = pert_fn(df[feature_name].values, **parameters)
        else:
            print(f"Warning: Feature {feature_name} could not be found to the dataset.")
    return df


def __apply_feature_maps(df: pd.DataFrame, config: dict):
    """Re-name the featues and their values according to the maps specified in the configuration file."""
    # for all the keys in the configuration file
    for key in config.keys():
        # if the key specifies a map
        if key[-4:] == "_Map":
            if key == "FeatureNames_Map":  # not for the feature names map
                continue
            feature_name = key[:-4]
            if feature_name in df.columns:
                df[feature_name] = df[feature_name].replace(config[key])
            else:
                print(f"Warning: {key} could not be matched to the dataset")
    # now the feature names map
    if "FeatureNames_Map" in config.keys():
        df = df.rename(columns=config["FeatureNames_Map"])
    return df


####################################################################################
# Generic dataset loading function
####################################################################################


def __load_dataset(
    csv_file: str, dataset_name: str, version=ORIGINAL_VERSION, seed=None
):
    """Generic dataset loading function. All the tranformations are specified in a yaml configuration file."""
    __validate_inputs(version)
    rng = np.random.default_rng(seed=seed)
    config = __load_yaml_config(dataset_name)

    # original
    df_original = utils.load_csv_df(csv_file)
    df_original = utils.strip_strings_in_dataframe(df_original)

    # move the target to the last column
    if "Target" in config.keys():
        df_original = move_column_to_position(
            df_original, config["Target"], len(df_original.columns) - 1
        )

    if version == ORIGINAL_VERSION:
        return df_original

    # perturbed
    df_perturbed = __apply_perturbations(df_original, config, seed=rng)
    if version == PERTURBED_VERSION:
        __check_perturbed_rows(df_original, df_perturbed)
        return df_perturbed

    # task
    df_task = __apply_feature_maps(df_perturbed, config)
    if version == TASK_VERSION:
        return df_task

    # statistical


def load_dataset(dataset_name: str, version=ORIGINAL_VERSION, *args, **kwargs):
    pass


####################################################################################
# Iris
####################################################################################


def load_iris(version=ORIGINAL_VERSION, seed=None):
    __validate_inputs(version)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df("iris.csv")
    if version == ORIGINAL_VERSION:
        return df_original
    X_data, y_data = df_original.iloc[:, :-1].values, df_original.iloc[:, -1].values

    # perturbed
    X_data_T = integer_perturbation(X_data * 10, size=3, seed=rng) / 10
    X_data_T[:, 3] = integer_perturbation(X_data[:, 3] * 10, size=1, seed=rng) / 10
    df_perturbed = pd.DataFrame(
        np.concatenate([X_data_T, y_data.reshape(-1, 1)], axis=1),
        columns=df_original.columns,
    )
    df_perturbed = permute_all_columns(df_perturbed, seed=rng)
    df_perturbed = move_column_to_position(df_perturbed, "species", 4)
    if version == PERTURBED_VERSION:
        __check_perturbed_rows(df_original, df_perturbed)
        return df_perturbed

    # task
    rename_map = {
        "sepal_length": "Measured Length of Sepal (cm)",
        "sepal_width": "Measured Width of Sepal (cm)",
        "petal_length": "Measured Length of Petal (cm)",
        "petal_width": "Measured Width of Petal (cm)",
        "species": "Kind of Flower",
    }
    df_task = df_perturbed.rename(columns=rename_map)
    rename_map = {
        "Iris-setosa": "Setosa",
        "Iris-virginica": "Virginica",
        "Iris-versicolor": "Versicolor",
    }
    df_task = df_task.replace(rename_map)
    df_task = add_normal_noise_and_round(df_task, noise_std=0.02, digits=2, seed=rng)

    if version == TASK_VERSION:
        return df_task

    # statistical
    X_data_S = statistical_transform(df_task.iloc[:, :-1].values, seed=seed)
    y_data_S = to_random_categorical(df_task.iloc[:, -1].values, seed=seed)

    df_statistical = pd.DataFrame(
        np.concatenate([X_data_S, y_data_S.reshape(-1, 1)], axis=1),
        columns=["X1", "X2", "X3", "X4", "Y"],
    )
    df_statistical["Y"] = df_statistical["Y"].astype(int)  # the label should be integer
    return df_statistical


####################################################################################
# Adult Income
####################################################################################


def load_adult(csv_file: str = "adult-train.csv", version=ORIGINAL_VERSION, seed=None):
    """The Adult Income dataset. http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html"""
    return __load_dataset(csv_file, "adult", version=version, seed=seed)


####################################################################################
# California Housing
####################################################################################


def load_housing(
    csv_file: str = "california-housing.csv", version=ORIGINAL_VERSION, seed=None
):
    __validate_inputs(version)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df(csv_file)
    if version == ORIGINAL_VERSION:
        return df_original

    # perturbed


####################################################################################
# Kaggle Titanic
####################################################################################


def load_titanic(csv_file: str, version=ORIGINAL_VERSION, seed=None):
    """csv_file: either the train or the test split of the datasets at https://www.kaggle.com/competitions/titanic"""
    __validate_inputs(version)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df(csv_file)
    if version == ORIGINAL_VERSION:
        return df_original


####################################################################################
# OpenML Diabetes
####################################################################################


def load_openml_diabetes(csv_file: str, version=ORIGINAL_VERSION, seed=None):
    __validate_inputs(version)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df(csv_file)
    if version == ORIGINAL_VERSION:
        return df_original

    # deepcopy of X_data
    X_data_T = np.array(df_original.values).copy()

    # for all women who had more than 3 pregnancies, randomly modify the number of pregnancies by 1
    indices = np.where(X_data_T[:, 0].astype(float) > 3)
    X_data_T[indices, 0] = (
        integer_perturbation(X_data_T[indices, 0].astype(float), 1)
        .astype(int)
        .astype(str)
    )

    # modify plasma glucose concentration by +/- 3 (respects the lower bound of 0 )
    X_data_T[:, 1] = (
        integer_perturbation(X_data_T[:, 1].astype(float), 3).astype(int).astype(str)
    )
    X_data_T[:, 2] = (
        integer_perturbation(X_data_T[:, 2].astype(float), 3).astype(int).astype(str)
    )
    X_data_T[:, 3] = (
        integer_perturbation(X_data_T[:, 3].astype(float), 3).astype(int).astype(str)
    )

    # respect many zeroes in insu
    indices = np.where(X_data_T[:, 4].astype(float) != 0)[0]
    X_data_T[indices, 4] = (
        integer_perturbation(X_data_T[indices, 4].astype(float), 3)
        .astype(int)
        .astype(str)
    )

    X_data_T[:, 5] = X_data_T[:, 5].astype(float) + np.random.normal(
        loc=0, scale=1, size=X_data_T.shape[0]
    )
    X_data_T[:, 5] = np.round(X_data_T[:, 5].astype(float), 1).astype(str)

    X_data_T[:, 6] = X_data_T[:, 6].astype(float) + np.random.normal(
        loc=0, scale=0.01, size=X_data_T.shape[0]
    )
    X_data_T[:, 6] = np.round(X_data_T[:, 6].astype(float), 3).astype(str)

    X_data_T[:, 7] = (
        integer_perturbation(X_data_T[:, 7].astype(float), 1).astype(int).astype(str)
    )

    X_data_T[0:5]


####################################################################################
# Perturbation Functions
####################################################################################


def integer_perturbation(
    x,
    size: int,
    respect_bounds: bool = True,
    frozen_indices=None,
    frozen_values=None,
    seed=None,
):
    """Perturb integer values. Does not modify nan values

        - does not modify nan values
        - does not change the min/max values in the data if respect bounds is true (default)
        - has the option not to change certain indices in the data (the frozen_indices parameter, a numpy array of indices)

    size: the maximum absolute size of the perturbation. Note that the perturbation is never zero (EXCEPT IF AT THE BOUNDARIES, todo change that)
    respect_bounds: if True, then the perturbed values are within the bounds of the original values

    perturbs only non-nan values.

    Returns: The perturbed array.
    """
    rng = np.random.default_rng(seed=seed)
    x = np.array(x.astype(int)).copy()
    # if x contains nan values, perturbe only the non-nan values
    if np.any(np.isnan(x)):
        x[~np.isnan(x)] = integer_perturbation(
            x[~np.isnan(x)],
            size=size,
            respect_bounds=respect_bounds,
            frozen_indices=frozen_indices,
            seed=seed,
        )
        return x
    # do not pertub frozen values
    if frozen_values is not None:
        # determine the indices of the frozen values, then go via the frozen_indices parameter
        if frozen_indices is None:
            frozen_indices = []
        frozen_indices.extend(np.argwhere(np.isin(x, frozen_values)))
        return integer_perturbation(
            x,
            size=size,
            respect_bounds=respect_bounds,
            frozen_indices=frozen_indices,
            frozen_values=None,
            seed=seed,
        )
    # do not pertub frozen indies
    if frozen_indices is not None:
        x_result = integer_perturbation(
            x, size=size, respect_bounds=respect_bounds, seed=seed
        )
        x_result[frozen_indices] = x[frozen_indices]
        return x_result
    # assert that x contains only integers
    assert np.all(np.equal(np.mod(x, 1), 0)), f"expected int values, found {x}"
    # apply the perturbation
    minimum = np.min(x)
    maximum = np.max(x)
    perturb = np.linspace(-size, size, 2 * size + 1)
    perturb = perturb[perturb != 0].astype(int)
    x = x + rng.choice(perturb, size=x.shape)
    if respect_bounds:
        x = np.maximum(x, minimum)
        x = np.minimum(x, maximum)
    return x


def value_perturbation(x: np.ndarray, size: int = 1, seed=None):
    """Replace each value either with the next smallest or next biggest. Requires numerical values.

    Returns: The perturbed array.
    """
    rng = np.random.default_rng(seed=seed)
    x = np.array(x).copy()
    res = np.zeros_like(x)
    unique_values = np.unique(x)  # this array is already sorted
    permute = np.linspace(-size, size, 2 * size + 1)
    permute = permute[permute != 0].astype(int)
    permute = rng.choice(permute, size=x.shape)
    for idx in range(len(x)):
        v = x[idx]
        v_idx = np.argwhere(unique_values == v)[0][0]
        new_v_idx = v_idx + permute[idx]
        new_v_idx = max(0, min(new_v_idx, len(unique_values) - 1))
        if new_v_idx == v_idx:  # guarantee a change in all values
            if v_idx == 0:
                new_v_idx = 1
            elif v_idx == len(unique_values) - 1:
                new_v_idx = len(unique_values) - 2
        res[idx] = unique_values[new_v_idx]
    return res


def add_normal_noise_and_round(df, noise_std=0.02, digits=2, seed=None):
    """
    Adds normal (Gaussian) noise to all float values in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be modified.
    std (float): Standard deviation of the noise.

    Returns:
    pd.DataFrame: A new DataFrame with noise added to float columns.
    """
    rng = np.random.default_rng(seed=seed)
    # Create a new DataFrame to avoid modifying the original one
    noisy_df = df.copy()

    # Iterate over each column in the DataFrame
    for col in noisy_df.columns:
        # Check if the column data type is numeric (not integer)
        if noisy_df[col].dtype == "float64":
            # Add normal noise to the column
            noise = rng.normal(0, noise_std, size=noisy_df[col].shape)
            noisy_df[col] += noise
            # round the column to two digits
            noisy_df[col] = np.round(noisy_df[col], digits)

    return noisy_df


def to_random_categorical(x: np.ndarray, seed=None):
    """Transform the input X to randomly chosen consecutive categorical values."""
    # random seed
    rng = np.random.default_rng(seed=seed)
    # to categorical
    unique_values = np.unique(x)
    unique_values = rng.permutation(unique_values)  # random permutation
    mapping = {value: i for i, value in enumerate(unique_values)}
    x = np.vectorize(mapping.get)(x)
    return x


def statistical_transform(
    X: np.ndarray, factor=-3.33, noise_std=0.05, decimals=2, seed=None
):
    """The statistical transform is a combination of the following steps:
    - scale to zero mean and unit variance
    - multiplily data by a constant factor (default: -3.33)
    - addition of normal noise with (default standard deviation: 0.05)
    - rounding (default: to two digits)
    - randomly permute the columns

    X: A numpy.ndarray with numeric values (a dataset).

    Returns: the transformed data (numpy.ndarray with the same shape as X)
    """
    assert X.ndim == 2, "X must be a 2D array"
    # random seed
    rng = np.random.default_rng(seed=seed)
    # zero mean and unit variance
    X_statistical = (X - X.mean(axis=0)) / X.std(axis=0)
    # transformation
    X_statistical = factor * X_statistical
    # add normal noise
    X_statistical += noise_std * rng.normal(size=X_statistical.shape)
    # round to two digits
    X_statistical = np.round(X_statistical, decimals=decimals)
    # permute the columns
    X_statistical = X_statistical[:, rng.permutation(X_statistical.shape[1])]
    return X_statistical


def permute_all_columns(df: pd.DataFrame, seed=None):
    """
    Randomly permutes all the columns of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are to be permuted.

    Returns:
    pd.DataFrame: A new DataFrame with permuted columns.
    """
    rng = np.random.default_rng(seed=seed)
    # Get a list of all columns
    columns = df.columns.tolist()

    # Randomly shuffle the columns
    rng.shuffle(columns)

    # Reindex the DataFrame with the new column order
    return df[columns]


def move_column_to_position(df: pd.DataFrame, column: str, position: int):
    """
    Moves a specified column in a DataFrame to a specified position.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    column (str): The name of the column to move.
    position (int): The new position for the column (0-based index).

    Returns:
    pd.DataFrame: A new DataFrame with the column moved to the specified position.
    """
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Ensure the position is within the valid range
    if position < 0 or position >= len(df.columns):
        raise ValueError("Position is out of range")

    # Remove the column and then insert it at the specified position
    col_data = df[column]
    df = df.drop(columns=[column])
    df.insert(position, column, col_data)

    return df
