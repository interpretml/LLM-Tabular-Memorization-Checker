import numpy as np
import pandas as pd

import importlib.resources as resources
import yaml

import tabmemcheck.utils as utils


ORIGINAL_TRANSFORM = "original"
PERTURBED_TRANSFORM = "perturbed"
TASK_TRANSFORM = "task"
STATISTICAL_TRANSFORM = "statistical"

DATASET_TRANSFORM = [
    ORIGINAL_TRANSFORM,
    PERTURBED_TRANSFORM,
    TASK_TRANSFORM,
    STATISTICAL_TRANSFORM,
]


def __validate_inputs(transform):
    assert transform in DATASET_TRANSFORM, (
        "dataset version must be one of "
        + ", ".join(DATASET_TRANSFORM)
        + f"got {transform}"
    )


####################################################################################
# Apply perturbations and transformations to a dataframe as specified
# in a YAML configuration file
####################################################################################


def __load_yaml_config(dataset_name: str):
    """Load from the resources folder of the package"""
    with resources.open_text(
        "tabmemcheck.resources.config", f"{dataset_name}.yaml"
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def __apply_perturbations(df: pd.DataFrame, config: dict, seed=None):
    """Perturb the values of the features according to the perturbations specified in the configuration file."""
    df = df.copy(deep=True)  # create a deep copy of the data frame
    PERTURBATION_METHODS = {
        "integer_perturbation": integer_perturbation,
        "float_perturbation": float_perturbation,
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


def __check_perturbed_rows(df_original, df_perturbed):
    df_common = pd.merge(df_original, df_perturbed, how="inner")
    if df_common.empty:
        print("None of the perturbed rows appear in the original dataset.")
    else:
        per_cent = 100.0 * df_common.shape[0] / df_perturbed.shape[0]
        print(
            f"{df_common.shape[0]} perturbed rows appear in the original dataset (that is {per_cent:.2f}% of all perturbed rows)."
        )


def __apply_transform(df: pd.DataFrame, config: dict, key):
    """Transform the values of the different features as specified in the config file."""
    df = df.copy(deep=True)  # create a deep copy of the data frame
    if (
        not key in config.keys()
    ):  # the user does not have to specify all the different transformations
        return df
    task_transform = config[key]  # a dict with the transformations
    TRANSFORMATION_METHODS = {
        "to_numeric": pd.to_numeric,
        "to_int": lambda x: x.apply(
            lambda x: np.NaN if pd.isna(x) or np.isinf(x) else int(x)
        ),
        "round": lambda x, decimals: x.round(decimals=decimals),
        "astype": lambda x, dtype: x.astype(dtype),
        "append": lambda x, value: x.astype(str) + value,
    }
    for feature_name in task_transform.keys():
        schedule = task_transform[feature_name]
        # check if the schedule is a list of lists
        if not isinstance(schedule[0], list):
            schedule = [schedule]
        for transform in schedule:
            method = transform[0]  # function name as string
            parameters = transform[1]  # dict with key-word arguments
            assert (
                method in TRANSFORMATION_METHODS.keys()
            ), f"Unknown task transform {method}."
            pert_fn = TRANSFORMATION_METHODS[method]
            df[feature_name] = pert_fn(df[feature_name], **parameters)
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
# Generic dataset loading function, with YAML configuration file
####################################################################################


def load_dataset(
    csv_file: str,
    dataset_name: str,
    transform=ORIGINAL_TRANSFORM,
    permute_columns=True,  # for perturbed transform
    seed=None,
):
    """Generic dataset loading function. All the tranformations are specified in a yaml configuration file."""
    __validate_inputs(transform)
    rng = np.random.default_rng(seed=seed)
    config = __load_yaml_config(dataset_name)

    # original
    df_original = utils.load_csv_df(csv_file, dtype=config.get("dTypes", None))
    df_original = utils.strip_strings_in_dataframe(df_original)

    if not "Target" in config.keys():  # assume that the target is the last column
        config["Target"] = df_original.columns[-1]

    # move the target to the last column
    df_original = move_column_to_position(
        df_original, config["Target"], len(df_original.columns) - 1
    )

    if transform == ORIGINAL_TRANSFORM:
        return df_original

    # perturbed
    df_perturbed = __apply_perturbations(df_original, config, seed=rng)
    if permute_columns:  # permute columns, but keep the target in the last column
        df_perturbed = permute_all_columns(df_perturbed, seed=rng)
        df_perturbed = move_column_to_position(
            df_perturbed, config["Target"], len(df_original.columns) - 1
        )

    if transform == PERTURBED_TRANSFORM:
        __check_perturbed_rows(df_original, df_perturbed)
        return df_perturbed

    # task
    df_task = __apply_transform(df_perturbed, config, key="TaskTransform")
    if (
        transform == TASK_TRANSFORM
    ):  # apply formatting that would cause problems for the statistical transform
        df_task = __apply_transform(df_task, config, key="TaskTransformOnly")
    df_task = __apply_feature_maps(df_task, config)
    if transform == TASK_TRANSFORM:
        return df_task

    # statistical
    categorical_features = df_task.select_dtypes(
        include="object"
    ).columns  # categorical features are those with dtype object

    # convert categorical features to integers
    for feature in categorical_features:
        df_task[feature] = to_categorical(
            df_task[feature].values, random=True, seed=rng
        )

    # to all columns that are not categorical, apply the statistical transform
    for feature in df_task.columns:
        if feature not in categorical_features:
            df_task[feature] = statistical_transform(
                df_task[feature].values.reshape(-1, 1), seed=rng
            )

    # rename columns to X1, X2, X3, ...
    df_statistical = df_task.rename(
        columns={feature: f"X{idx+1}" for idx, feature in enumerate(df_task.columns)}
    )
    # the last column is the target, rename it to Y
    df_statistical = df_statistical.rename(columns={df_statistical.columns[-1]: "Y"})

    return df_statistical


####################################################################################
# Iris
####################################################################################


def load_iris(transform=ORIGINAL_TRANSFORM, seed=None):
    __validate_inputs(transform)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df("iris.csv")
    if transform == ORIGINAL_TRANSFORM:
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
    if transform == PERTURBED_TRANSFORM:
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

    if transform == TASK_TRANSFORM:
        return df_task

    # statistical
    X_data_S = statistical_transform(df_task.iloc[:, :-1].values, seed=seed)
    y_data_S = to_categorical(df_task.iloc[:, -1].values, random=True, seed=seed)

    df_statistical = pd.DataFrame(
        np.concatenate([X_data_S, y_data_S.reshape(-1, 1)], axis=1),
        columns=["X1", "X2", "X3", "X4", "Y"],
    )
    df_statistical["Y"] = df_statistical["Y"].astype(int)  # the label should be integer
    return df_statistical


####################################################################################
# Adult Income
####################################################################################


def load_adult(csv_file: str = "adult-train.csv", *args, **kwargs):
    """The Adult Income dataset. http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html"""
    return load_dataset(csv_file, "adult", *args, **kwargs)


####################################################################################
# California Housing
####################################################################################


def load_housing(csv_file: str, *args, **kwargs):
    return load_dataset(csv_file, "housing", *args, **kwargs)


####################################################################################
# Kaggle Titanic
####################################################################################


def load_titanic(csv_file: str, transform=ORIGINAL_TRANSFORM, seed=None):
    """csv_file: either the train or the test split of the datasets at https://www.kaggle.com/competitions/titanic"""
    __validate_inputs(transform)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df(csv_file)
    if transform == ORIGINAL_TRANSFORM:
        return df_original


####################################################################################
# OpenML Diabetes
####################################################################################


def load_openml_diabetes(csv_file: str, transform=ORIGINAL_TRANSFORM, seed=None):
    __validate_inputs(transform)
    rng = np.random.default_rng(seed=seed)

    # original
    df_original = utils.load_csv_df(csv_file)
    if transform == ORIGINAL_TRANSFORM:
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
# Function for perturbation and transformation of data
####################################################################################


def numeric_perturbation(
    X: np.ndarray,
    perturbation_matrix: np.ndarray,
    respect_bounds: bool = True,
    frozen_values=None,
    frozen_indices=None,
):
    """Perturb a np.ndarray of numeric values using the perturbations in perturbation_matrix.

    The pertubation is done in the following way:

        - Does not perturb nan values.
        - If respect bounds is true (default), does not perturb beyond the min/max values in the dat.
        - Does not perturb frozen values (if specified) and frozen indices (if specified).

    Returns: The perturbed array.
    """
    # assert that x contains only numeric values
    assert np.issubdtype(
        X.dtype, np.number
    ), f"Expected numeric values, found {X.dtype}."
    # do not pertub frozen values
    if frozen_values is not None:
        # determine the indices of the frozen values, then go via the frozen_indices parameter
        if frozen_indices is None:
            frozen_indices = []
        frozen_indices.extend(np.argwhere(np.isin(X, frozen_values)))
        return numeric_perturbation(
            X,
            perturbation_matrix=perturbation_matrix,
            respect_bounds=respect_bounds,
            frozen_indices=frozen_indices,
            frozen_values=None,
        )
    # do not pertub frozen indies
    if frozen_indices is not None:
        x_result = numeric_perturbation(
            X, perturbation_matrix=perturbation_matrix, respect_bounds=respect_bounds
        )
        x_result[frozen_indices] = X[frozen_indices]
        return x_result
    # if x contains nan values, perturbe only the non-nan values
    if np.any(np.isnan(X)):
        X[~np.isnan(X)] = numeric_perturbation(
            X[~np.isnan(X)],
            perturbation_matrix=perturbation_matrix[~np.isnan(X)],
            respect_bounds=respect_bounds,
            frozen_indices=None,  # frozen values and indices have already been taken care of at this point
            frozen_values=None,
        )
        return X
    # store min/max
    minimum = np.min(X)
    maximum = np.max(X)
    # apply the perturbation
    X = X + perturbation_matrix
    # respect the bounds
    if respect_bounds:
        X = np.maximum(X, minimum)
        X = np.minimum(X, maximum)
    return X


def integer_perturbation(
    X: np.ndarray,
    size: int,
    respect_bounds: bool = True,
    frozen_indices=None,
    frozen_values=None,
    seed=None,
):
    """Perturb integer values with values in the range [-size, size], but never zero (except if at the boundaries).

    Returns: The perturbed array (an array of integers).
    """
    # assert that x does not have any significant digits after the decimal point
    assert np.all(np.equal(np.mod(X, 1), 0)), f"Expected integer values, found {X}."
    # convert x to integer
    X = np.array(X.astype(int)).copy()
    # generate the perturbation matrix
    rng = np.random.default_rng(seed=seed)
    perturb = np.linspace(-size, size, 2 * size + 1)
    perturb = perturb[perturb != 0].astype(int)
    perturbation_matrix = rng.choice(perturb, size=X.shape)
    # apply the perturbation
    return numeric_perturbation(
        X,
        perturbation_matrix,
        respect_bounds=respect_bounds,
        frozen_indices=frozen_indices,
        frozen_values=frozen_values,
    )


def float_perturbation(
    X: np.ndarray,
    size: float,
    respect_bounds: bool = True,
    frozen_indices=None,
    frozen_values=None,
    seed=None,
):
    """Perturb numeric values with values in the range [-size, size], but never zero (except if at the boundaries).

    Returns: The perturbed array (an array of floats).
    """
    # assert that x contains only numeric values
    assert np.issubdtype(
        X.dtype, np.number
    ), f"Expected numeric values, found {X.dtype}."
    # determine the relevant scaling factor
    scaling_factor = 1.0
    while scaling_factor * size % 1 != 0:
        scaling_factor *= 10.0
    size = int(scaling_factor * size)
    # generate the perturbation matrix
    rng = np.random.default_rng(seed=seed)
    perturb = np.linspace(-size, size, 2 * size + 1)
    perturb = perturb[perturb != 0]
    perturbation_matrix = rng.choice(perturb, size=X.shape) / scaling_factor
    # apply the perturbation
    return numeric_perturbation(
        X,
        perturbation_matrix,
        respect_bounds=respect_bounds,
        frozen_indices=frozen_indices,
        frozen_values=frozen_values,
    )


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


def to_categorical(x: np.ndarray, random=False, seed=None):
    """Transform the input X to randomly chosen categorical values."""
    # random seed
    rng = np.random.default_rng(seed=seed)
    # to categorical
    unique_values = np.unique(x)
    if random:  # assign randomly permutated values
        unique_values = rng.permutation(unique_values)
    mapping = {value: i for i, value in enumerate(unique_values)}
    x = np.vectorize(mapping.get)(x)
    return x


def statistical_transform(
    X: np.ndarray,
    factor=-3.33,
    noise_std=0.05,
    decimals=2,
    seed=None,
):
    """The statistical transform is a combination of the following steps:
    - scale to zero mean and unit variance
    - multiplily data by a constant factor (default: -3.33)
    - addition of normal noise with (default standard deviation: 0.05)
    - rounding (default: to two digits)

    X: A numpy.ndarray with numeric values (a dataset).

    Returns: the transformed data (numpy.ndarray with the same shape as X)
    """
    assert X.ndim == 2, "X must be a 2D array"
    # assert that X contains only numeric values
    assert np.issubdtype(
        X.dtype, np.number
    ), f"expected numeric values, found {X.dtype}"
    # random seed
    rng = np.random.default_rng(seed=seed)
    # zero mean and unit variance
    X_statistical = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    # transformation
    X_statistical = factor * X_statistical
    # add normal noise
    X_statistical += noise_std * rng.normal(size=X_statistical.shape)
    # round to two digits
    X_statistical = np.round(X_statistical, decimals=decimals)
    # permute the columns
    # X_statistical = X_statistical[:, rng.permutation(X_statistical.shape[1])]
    return X_statistical


####################################################################################
# Other functions to transform and organize the data
####################################################################################


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
