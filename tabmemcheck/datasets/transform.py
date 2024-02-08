####################################################################################
# This module contains various function to perturb, organize and transform data.
# Some of the functions apply to a numpy array, others to a pandas DataFrame.
####################################################################################

import numpy as np
import pandas as pd


####################################################################################
# numpy array
####################################################################################


def numeric_perturbation(
    X: np.ndarray,
    perturbation_matrix: np.ndarray,
    respect_bounds: bool = True,
    frozen_values=None,
    frozen_indices=None,
    ignore_object=False,
):
    """Perturb a np.ndarray X of numeric values using the perturbations in perturbation_matrix.

    X and perturbation_matrix must have the same shape.

    The pertubation respects the following boundary conditions:
        - Does not perturb nan values.
        - Does not perturb frozen values (if specified) and frozen indices (if specified).
        - If respect bounds is true (default), does not perturb beyond the min/max values in the data.

    Returns: The perturbed array.
    """
    # if ignore_object is True, only perturb values that can be converted to float
    if ignore_object:
        numeric_indices = []
        for index, value in np.ndenumerate(X):
            try:
                # Attempt to convert value to float
                float(value)
                # If successful, append the index to the list
                numeric_indices.append(index)
            except ValueError:
                continue
                # continue
        X[numeric_indices] = numeric_perturbation(
            X[numeric_indices].astype(np.number),
            perturbation_matrix=perturbation_matrix[numeric_indices],
            respect_bounds=respect_bounds,
            frozen_indices=frozen_indices,
            frozen_values=frozen_values,
            ignore_object=False,
        )
        return X
    # assert that x contains only numeric values
    assert np.issubdtype(
        X.dtype, np.number
    ), f"Expected numeric values, found {X.dtype}."
    # do not pertub frozen values
    if frozen_values is not None:
        # determine the indices of the frozen values, then go via the frozen_indices parameter
        if frozen_indices is None:
            frozen_indices = []
        frozen_values = np.array(frozen_values).astype(X.dtype).flatten()
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
            X.copy(),  # create a deep copy to avoid modifying the original array
            perturbation_matrix=perturbation_matrix,
            respect_bounds=respect_bounds,
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
    scale: float = 1e0,
    respect_bounds: bool = True,
    frozen_indices=None,
    frozen_values=None,
    ignore_object=False,
    seed=None,
):
    """Perturb with integer values from the range [-size, size], but never zero (zero perturbation can still occur due to a boundary condition).

    The perturbation is scaled with the parameter scale. This allows to apply perturbations at specific decimal positions.

    Returns: The perturbed array.
    """

    # TODO validate the position parameter
    # assert that x does not have any significant digits after the decimal point
    # assert np.all(np.equal(np.mod(X, 1), 0)), f"Expected integer values, found {X}."
    # TODO re-introduce if required? with additional parameter perhaps
    # convert x to integer
    # X = np.array(X.astype(int)).copy()
    # generate the perturbation matrix
    rng = np.random.default_rng(seed=seed)
    perturb = np.linspace(-size, size, 2 * size + 1)
    perturb = perturb[perturb != 0].astype(int)
    perturbation_matrix = rng.choice(perturb, size=X.shape)
    # modify the perturbation matrix to respect the decimal position / scale
    if int(scale) == scale:
        scale = int(scale)
    perturbation_matrix = perturbation_matrix * scale
    # apply the perturbation
    return numeric_perturbation(
        X,
        perturbation_matrix,
        respect_bounds=respect_bounds,
        frozen_indices=frozen_indices,
        frozen_values=frozen_values,
        ignore_object=ignore_object,
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


def swap_perturbation(x: np.ndarray, size: int = 1, seed=None):
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


def add_normal_noise_and_round_array(
    X: np.array,
    noise_std=0.02,
    digits=2,
    respect_bounds=True,
    frozen_indices=None,
    frozen_values=None,
    seed=None,
):
    """
    Adds normal (Gaussian) noise to all float values in a numpy array.

    Parameters:
    X (np.ndarray): The array to be modified.
    std (float): Standard deviation of the noise.

    Returns:
    np.ndarray: A new array with noise added to float columns.
    """
    rng = np.random.default_rng(seed=seed)
    # Create a new array to avoid modifying the original one
    # apply noise
    noisy_X = numeric_perturbation(
        X,
        rng.normal(0, noise_std, size=X.shape),
        respect_bounds=respect_bounds,
        frozen_indices=frozen_indices,
        frozen_values=frozen_values,
    )
    # round the array to the specified number of digits
    return np.round(noisy_X, digits)


def to_categorical(x: np.ndarray, random=False, seed=None):
    """Transform the input X to categorical values."""
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
# data frame
####################################################################################


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
