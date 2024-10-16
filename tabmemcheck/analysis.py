#
# functions to analyze model responses
#
# these functions are separate from the test_functions and completions.
#
# the reason for this is that it allos us to store the responses of the tests in csv files and perform (more) analysis later.
#


import numpy as np
import pandas as pd

from scipy.stats import ttest_ind

import tabmemcheck.utils as utils


#################################################################
# test if a response or row is contained in a data frame
#################################################################


def string_strip(x):
    """Convert the input (dataframe, series, or string) to string and strip trailing whitespaces."""
    # if x is data frame
    if isinstance(x, pd.DataFrame):
        x = x.astype(str)
        x = x.map(lambda x: x.strip())
    elif isinstance(x, pd.Series):
        x = x.astype(str)
        x = x.apply(lambda x: x.strip())
    elif isinstance(x, str):
        x = x.strip()
    else:
        raise ValueError(f"Unkown type for string_strip: {type(x)}.")
    return x


def validate_partial_row(x, feature_names):
    """check if the input is a row with a SUBSET of the features in feature_names"""
    # if x is a string, parse it
    if isinstance(x, str):
        x = utils.parse_feature_string(x, feature_names)
    # if x is a series, convert it to a dataframe
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x).T
    # at this point, x should be a dataframe with a single row. check this.
    assert isinstance(x, pd.DataFrame)
    assert x.shape[0] == 1
    # x should only contain feature names from the specified feature_names
    assert set(x.columns).issubset(
        set(feature_names)
    ), f"Invalid feature names in x: {set(feature_names) - set(x.columns)}."
    return x


def find_matches(
    df: pd.DataFrame,
    x,
    string_dist_fn=utils.levenshtein_distances,
    match_floating_point=True,
    strip_quotation_marks=True,
):
    """Find the closest matches between a row x and all rows in the dataframe df. By default, we use the levenshtein distance as the distance metric.

    This function can handle some formatting differences between the values in the original data and LLM responses that should still be counted as equal.

    :param df: a pandas dataframe.
    :param x: a string, a pandas dataframe or a pandas Series.
    :param string_dist_fn: a function that computes the distance between two strings. By default, this is the levenshtein distance.
    :param match_floating_point: if True, handes floating point formatting differences, e.g. 0.28 vs. .280 or 172 vs 172.0 (default: True).
    :param strip_quotation_marks: if True, strips quotation marks from the values in df and x (to handle the case where a model responds with "23853", and the value in the data is 23853) (default: True).

    :return: the minimum distance and the matching rows in df.
    """
    # x should be a dataframe with a single row, or be convertible to this format
    x = validate_partial_row(x, df.columns)
    # create a deep copy of df
    df = df.copy(deep=True)
    # convert to string & strip both df and x of any leading and trailing whitespaces. this is very important!
    df = string_strip(df)
    x = string_strip(x)
    # optionall, also strip quotation marks
    if strip_quotation_marks:
        df = df.map(lambda x: x.strip('"'))
        x = x.map(lambda x: x.strip('"'))
    # for all the features that are present in x, compute the levenshtein distance between the feature value and the respective feature values in df
    D = np.zeros(df.shape[0])
    for feature_name in df.columns:
        if feature_name in x.columns:
            D_feature = np.array(
                string_dist_fn(x[feature_name].values[0], df[feature_name].tolist())
            )
            # can the feature value be converted to a floating point number?
            if match_floating_point:
                try:
                    x_value_float = float(x[feature_name].values[0])
                    # at this point, x_value_float is a valid floating point number
                    df_column_float = pd.to_numeric(
                        df[feature_name], errors="coerce"
                    ).tolist()
                    D_feature_float = np.abs(
                        x_value_float - np.array(df_column_float).flatten()
                    )
                    D_feature_float[D_feature_float > 0] = np.inf
                    D_feature_float[np.isnan(D_feature_float)] = (
                        np.inf
                    )  # do not propagate NaNs
                    D_feature = np.minimum(D_feature, D_feature_float)
                except:
                    pass
            # print(feature_name, D_feature)
            D += D_feature
    min_dist = np.min(D)
    return min_dist, df[D == min_dist]


def is_in_df(df, x):
    min_dist, matches = find_matches(df, x, utils.strings_unequal)
    if min_dist == 0:
        print(matches)
    return min_dist == 0


#################################################################
# callbacks. these are factories that return the callback functions
#################################################################


def csv_match_callback(csv_file):
    """given a response from the LLM, check if it occurs in the dataset."""
    # load the csv file
    df = utils.load_csv_df(csv_file)

    def callback(messages, response):
        result = ""
        # does the last message contain conditional features? (i.e. conditional sampling)
        MAGIC_STRING = "Feature Values: "
        query = messages[-1]["content"]
        pos = query.find(MAGIC_STRING)
        cond_df = None
        if pos != -1:
            query = query[pos + len(MAGIC_STRING) :]
            # parse the query
            cond_df = utils.parse_feature_string(query, df.columns)
        # parse the response
        response_df = utils.parse_feature_string(response, df.columns)
        # join the conditional and response dataframes
        if cond_df is not None:
            response_df = pd.concat([cond_df, response_df], axis=1)
        # check if the response is in the dataframe
        dist, matches = find_matches(df, response_df)
        if dist == 0:
            result = f"Exact match with row {matches.index.tolist()[0]}."
        else:
            result = f"Distance {dist} to row {matches.index.tolist()[0]}."
        return result

    return callback


#################################################################
# conditional completion analysis
#################################################################


def conditional_completion_analysis(csv_file, completions_df):
    """Analysis for the conditional completion test"""
    data_df = utils.load_csv_df(csv_file)
    feature_names = utils.get_feature_names(csv_file)

    # the unique values of 'num_prefix_features'
    num_prefix_features = completions_df["num_prefix_features"].unique()

    # for each number of prefix features
    for num_prefix_feature in num_prefix_features:
        completion_feature_name = feature_names[num_prefix_feature]
        marginal_distribution = data_df[completion_feature_name].values
        valid_completions = []
        valid_marginal_completios = []

        # the respective data frame with the responses
        df = completions_df[completions_df["num_prefix_features"] == num_prefix_feature]

        # for each response in the data frame
        for _, row in df.iterrows():
            # look at the response up to num_prefix_featues +1 (that is, inlcuding the first completed feature)
            # does the response occur in the dataset?
            response = row[: num_prefix_feature + 1]
            # print(response)
            # print(type(response))
            valid_completions.append(is_in_df(data_df, response))

            # now, replace the actual completion from a completion drawn from the marginal distribution in the dataset
            response[completion_feature_name] = np.random.choice(marginal_distribution)
            valid_marginal_completios.append(is_in_df(data_df, response))

        print(np.mean(valid_completions))
        print(np.mean(valid_marginal_completios))


#################################################################
# old functions here, maybe move at some point
#################################################################


def levenshtein_distance_t_test(x, y, z, alternative="two-sided", return_dist=False):
    """Test whether x is closer to y than z in Levenshtein distance using a t-test.

    :param x, y, z: a list of strings.
    :param alternative: the alternative hypothesis, either 'two-sided', 'greater', or 'less'.
    :return_dist: if True also return the distances between x and y, and x and z.
    :return: scipy.stats._result_classes.TtestResult"""
    # convert numpy arrays to lists
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(y, np.ndarray):
        y = y.tolist()
    if isinstance(z, np.ndarray):
        z = z.tolist()
    # now check that all inputs are lists
    for input in [x, y, z]:
        if not isinstance(input, list):
            raise ValueError(
                "Invalid input types for levenshtein_distance_test: {}".format(
                    type(input)
                )
            )
    # all lists must have the same length
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError(
            "Invalid input lengths for levenshtein_distance_test: {}, {}, {}".format(
                len(x), len(y), len(z)
            )
        )
    # compute distances and compute mean in case y or z are lists of lists
    dist_x_y = utils.levenshtein_distances(x, y)
    dist_x_y = [np.mean(x) for x in dist_x_y]
    dist_x_z = utils.levenshtein_distances(x, z)
    dist_x_z = [np.mean(x) for x in dist_x_z]
    # print the means and standard deviations (debugging)
    # print("mean x-y: {}, std x-y: {}".format(np.mean(dist_x_y), np.std(dist_x_y)))
    # print("mean x-z: {}, std x-z: {}".format(np.mean(dist_x_z), np.std(dist_x_z)))
    # the t-test statistic
    test_statistic = ttest_ind(
        dist_x_y, dist_x_z, equal_var=False, alternative=alternative
    )
    # return results
    if return_dist:
        return test_statistic, dist_x_y, dist_x_z
    return test_statistic


def build_first_token(csv_file, verbose=False):
    """Given a csv file, build a first token that can be used in the first token test.

    The first token is constructed by taking the first n digits of every row in the csv file (that is, this functions determines the n).
    Using the first n digits improves upon using the first digit on datasets where the first digit is always the same or contains few distinct values.

    Note: This function does NOT check if the constructed first token is random.

    :param csv_file: the path to the csv file.
    :param verbose: if True, print the first tokens and their counts.
    :return: the number of digits that make up the first token.
    """
    csv_rows = utils.load_csv_rows(csv_file, header=False)
    num_rows = len(csv_rows)
    for num_digits in range(1, 7):
        tokens = [x[:num_digits] for x in csv_rows]
        # count the occurences of each token
        values, counts = np.unique(tokens, return_counts=True)
        # we have found good first tokens if 1) there are at least 3 different tokens
        #                                    2) the most common token appears in at most 50% of the rows
        if len(counts) >= 3 and np.max(counts) < 0.5 * num_rows:
            if verbose:
                print("First tokens:", list(zip(values, counts)))
            return num_digits
    raise ValueError("Failed to construct valid first tokens.")


def find_most_unique_feature(csv_file):
    """Given a csv file, find the feature that has the most unique values. This is the default feature used for the feature completion test.

    :param csv_file: the path to the csv file.
    :return: the name of the most unique feature and the fraction of unique values.
    """
    feature_names = utils.get_feature_names(csv_file)
    df = utils.load_csv_df(csv_file)
    df = string_strip(df)
    # for each feature, compute the number of unique values
    num_unique_values = [
        len(df[feature_name].unique()) for feature_name in feature_names
    ]
    # the most unique feature
    most_unique_feature = feature_names[np.argmax(num_unique_values)]
    # the fraction of unique values
    frac_unique_values = np.max(num_unique_values) / df.shape[0]
    return most_unique_feature, frac_unique_values
