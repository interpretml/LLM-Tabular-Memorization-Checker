import os

import numpy as np
import pandas as pd

import jellyfish
import difflib
import tempfile
import itertools

import csv

from scipy.stats import bootstrap
from sklearn import metrics

import importlib

import tabmemcheck as tabmem

#################################################################
# the package ships with a number of csv files,
# used for zero-shot prompting.
#################################################################
from contextlib import contextmanager


@contextmanager
def plain_context(x):
    yield x


CSV_FILE_RESOURCES = "CSV_FILE_RESOURCES"


def _init_resource_csv_files():
    # load the csv files that are available as resources
    global CSV_FILE_RESOURCES
    if not CSV_FILE_RESOURCES in tabmem.config:
        csv_files = [
            x
            for x in importlib.resources.contents("tabmemcheck.resources.csv")
            if x.endswith(".csv")
        ]
        tabmem.config[CSV_FILE_RESOURCES] = csv_files


def _csv_file(csv_file):
    """Provides a contex manager for csv files."""
    global CSV_FILE_RESOURCES
    # if the csv file does not exist on the files system, check if it's available as a resource
    if not os.path.exists(csv_file):
        _init_resource_csv_files()
        if csv_file in tabmem.config[CSV_FILE_RESOURCES]:
            return importlib.resources.path("tabmemcheck.resources.csv", csv_file)
    return plain_context(csv_file)


#################################################################
# basic utilities for csv files
#################################################################


def get_dataset_name(csv_file):
    """Returns the name of the dataset."""
    return os.path.splitext(os.path.basename(csv_file))[0]


def get_delimiter(csv_file):
    """Returns the delimiter of a csv file."""
    with _csv_file(csv_file) as csv_file:
        sniffer = csv.Sniffer()
        with open(csv_file) as fp:
            delimiter = sniffer.sniff(fp.read(5000)).delimiter
        return delimiter


def get_feature_names(csv_file):
    """Returns the names of the features in a csv file (a list of strings)."""
    with _csv_file(csv_file) as csv_file:
        df = load_csv_df(csv_file)
        return df.columns.tolist()


CSV_MAX_ROWS_WARNING_PRINTED = False
def load_csv_df(csv_file, header=True, delimiter="auto", **kwargs):
    global CSV_MAX_ROWS_WARNING_PRINTED
    """Load a csv file as a pandas data frame."""
    with _csv_file(csv_file) as csv_file:
        # auto detect the delimiter from the csv file
        if delimiter == "auto":
            delimiter = get_delimiter(csv_file)
        # load the csv file
        max_rows = tabmem.config.csv_max_rows
        df = pd.read_csv(csv_file, delimiter=delimiter, nrows=max_rows+1, **kwargs)
        # Check if the file has more rows than n
        if len(df) > max_rows and not CSV_MAX_ROWS_WARNING_PRINTED:
            print(f'Info: Found a CSV file with more than {max_rows} rows. Note that tabmemcheck is configured to use only the first {max_rows} rows. Set tabmemcheck.config.csv_max_rows to change this behavior.')
            CSV_MAX_ROWS_WARNING_PRINTED = True
        # Truncate the dataframe to the first n rows
        df = df.head(max_rows)
        # optionally, remove the header
        if not header:
            df = df.iloc[1:]
        return df


def load_csv_rows(csv_file, header=True):
    """Load a csv file as a list of strings, with one string per row."""
    global CSV_MAX_ROWS_WARNING_PRINTED
    with _csv_file(csv_file) as csv_file:
        with open(csv_file, "r") as f:
            data = list(itertools.islice(f, tabmem.config.csv_max_rows+1))
        # check if the file has more rows than n, if yes print warning and reduce to n
        if len(data) > tabmem.config.csv_max_rows and not CSV_MAX_ROWS_WARNING_PRINTED:
            print(f'Info: Found a CSV file with more than {tabmem.config.csv_max_rows} rows. Note that tabmemcheck is configured to use only the first {tabmem.config.csv_max_rows} rows. Set tabmemcheck.config.csv_max_rows to change this behavior.')
            CSV_MAX_ROWS_WARNING_PRINTED = True
        data = data[:tabmem.config.csv_max_rows]
        # remove all trailing newlines
        data = [line.rstrip("\n") for line in data]
        # remove all empty rows
        data = [line for line in data if len(line) > 0]
        # optionally, remove the header
        if not header:
            data = data[1:]
        return data


def load_csv_string(csv_file, header=True, size=10000000):
    """Load a csv file as a single string."""
    with _csv_file(csv_file) as csv_file:
        # load the csv file into a single string
        with open(csv_file, "r") as f:
            data = f.read(size)
        # remove header TODO, this currently only works if header does not contain "\n"
        if not header:
            data = data.split("\n")[1:]
            data = "\n".join(data)
        return data


def load_csv_array(csv_file, add_feature_names=False):
    """Load a csv file as a 2d numpy array where each entry is a string.

    :add_feature_names: if true, then each entry will have the format "feature_name = feature_value"
    :returns: a 2d numpy array of strings
    """
    with _csv_file(csv_file) as csv_file:
        # load csv as a pandas dataframe
        df = load_csv_df(csv_file)
        feature_names = get_feature_names(csv_file)
        # convert all the entries to strings
        df = df.astype(str)
        # strip whitespaces at beginning and end
        df = df.map(lambda x: x.strip())
        # if add_feature_names is true, then convert each entry to the format "feature_name = feature_value"
        if add_feature_names:
            for feature_name in feature_names:
                df[feature_name] = feature_name + " = " + df[feature_name]
        # the underlying numpy array
        data = df.values
        return data


def tmp_csv_file(df, dataset_name):
    """create a temporary csv file from a pandas dataframe.

    Returns: the path of the file (string)
    """
    # create a temporary folder for our dataset
    tmp_folder = tempfile.mkdtemp()
    # we save the pandas dataframe in the temporary folder, using the name of the dataset
    file_path = os.path.join(tmp_folder, f"{dataset_name}.csv")
    df.to_csv(file_path, index=False)
    return file_path


#################################################################
# more advanced functions for csv files
# we directly construct data for prompts
#################################################################


def load_samples(csv_file, add_feature_names=True):
    """Load a csv file as a list of ''Feature name = Feature value'' strings.

    :returns: description, samples
    """
    # load the relevant information from the csv file
    dataset_name = get_dataset_name(csv_file)
    feature_names = get_feature_names(csv_file)
    X = load_csv_array(csv_file, add_feature_names=add_feature_names)
    description = f"Dataset: {dataset_name}\nFeature Names: " + ", ".join(feature_names)
    samples = [", ".join(x) for x in X]
    return description, samples


def load_cond_samples(
    csv_file, cond_feature_names, add_description=True, add_feature_names=True
):
    """Returns: prefixes, suffixes

      Prefixes:
      =======
    ['Dataset: adult
      Feature Names: Age, WorkClass, fnlwgt, Education, EducationNum, MaritalStatus, Occupation, Relationship, Race, Gender, CapitalGain, CapitalLoss, HoursPerWeek, NativeCountry, Income
      Feature Values: Age = 39, WorkClass = <=50K', ...]

      Samples:
      ========
      ['fnlwgt = 1231, .., Income = <=50K', ...]
    """
    # load the relevant information from the csv file
    dataset_name = get_dataset_name(csv_file)
    feature_names = get_feature_names(csv_file)
    # assert that all cond feature name are valid
    assert all(
        [cond_feature_name in feature_names for cond_feature_name in cond_feature_names]
    ), "Invalid conditional feature names."
    X = load_csv_array(csv_file, add_feature_names=add_feature_names)
    cond_feature_indices = [feature_names.index(name) for name in cond_feature_names]
    sample_feature_indices = [
        idx for idx in range(len(feature_names)) if idx not in cond_feature_indices
    ]
    description = f"Dataset: {dataset_name}\nFeature Names: " + ", ".join(feature_names)
    # prefixes include the cond_feature_names
    prefixes = [", ".join(x) for x in X[:, cond_feature_indices]]
    if add_description:
        prefixes = [description + "\nFeature Values: " + prefix for prefix in prefixes]
    suffixes = [", ".join(x) for x in X[:, sample_feature_indices]]
    return prefixes, suffixes


def load_cond_samples_target(
    csv_file,
    cond_feature_names,
    target=None,
    add_description=True,
    add_feature_names=True,
):
    """Returns: prefixes, suffixes

    Prefixes:
    =======
    ['Dataset: adult
    Feature Names: Age, WorkClass, fnlwgt, Education, EducationNum, MaritalStatus, Occupation, Relationship, Race, Gender, CapitalGain, CapitalLoss, HoursPerWeek, NativeCountry, Income
    Feature Values: Age = 39, WorkClass = <=50K', ...,
    Target: Income]

    Samples:
    ========
    ['fnlwgt = 1231, .., Income = <=50K', ...]
    """
    # load the relevant information from the csv file
    dataset_name = get_dataset_name(csv_file)
    feature_names = get_feature_names(csv_file)
    # assert that all cond feature name are valid
    assert all(
        [cond_feature_name in feature_names for cond_feature_name in cond_feature_names]
    ), "Invalid conditional feature names."
    # assert that the target is valid
    if target is not None:
        assert target in feature_names, "Invalid target."
    X = load_csv_array(csv_file, add_feature_names=add_feature_names)
    cond_feature_indices = [feature_names.index(name) for name in cond_feature_names]
    sample_feature_indices = [
        idx for idx in range(len(feature_names)) if idx not in cond_feature_indices
    ]
    description = f"Dataset: {dataset_name}\nFeature Names: " + ", ".join(feature_names)
    # prefixes include the cond_feature_names
    prefixes = [", ".join(x) for x in X[:, cond_feature_indices]]
    if add_description:
        prefixes = [description + "\nFeature Values: " + prefix for prefix in prefixes]
    if target is not None:
        prefixes = [prefix + "\nTarget: " + target for prefix in prefixes]
    if target is None:
        suffixes = [", ".join(x) for x in X[:, sample_feature_indices]]
    else:
        target_index = feature_names.index(target)
        suffixes = [x[target_index] for x in X]
    return prefixes, suffixes


#################################################################
# parsing of model responses
#################################################################


def parse_feature_string(
    s, feature_names, as_dict=False, in_list=False, final_delimiter=","
):
    """Parse a string (model response) of the form "feature_name = feature_value, feature_name = feature_value, ..." into a pandas dataframe."""
    feature_dict = {}
    # we use the magic strings 'feature_name = '
    magic_strings = [name + " = " for name in feature_names]
    while len(s) > 3:
        # find the next magic string
        indices = [s.find(magic_string) for magic_string in magic_strings]
        if max(indices) == -1:
            break
        start = min(index for index in indices if index > -1)
        feature_index = indices.index(start)
        # is there any magic string after that?
        next_indices = [
            s.find(magic_string, start + 3) for magic_string in magic_strings
        ]
        # if there is, then the value is between the two magic strings
        if max(next_indices) != -1:
            end = min(index for index in next_indices if index > -1)
            feature_dict[feature_names[feature_index]] = [
                s[
                    start + len(magic_strings[feature_index]) : s[:end].rfind(",")
                ].strip()
            ]
        # if there isn't, then the value is between the magic string and the final delimitier, or the end of the string
        else:
            end = -1
            if final_delimiter is not None:
                end = s[start + len(magic_strings[feature_index]) :].find(
                    final_delimiter
                )
            if end > -1:
                end = start + len(magic_strings[feature_index]) + end
            else:
                end = len(s)
            feature_dict[feature_names[feature_index]] = [
                s[start + len(magic_strings[feature_index]) : end].strip()
            ]
        s = s[end:]
    if not in_list:
        for key in feature_dict:
            feature_dict[key] = feature_dict[key][0]
    if as_dict:
        return feature_dict
    return pd.DataFrame(feature_dict, index=[0])


def parse_feature_stings(strings, feature_names, **kwargs):
    """Parse a list of features strings into a pandas dataframe."""
    parsed = []
    for s in strings:
        s_df = parse_feature_string(s, feature_names, **kwargs)
        parsed.append(s_df)
    return pd.concat(parsed, ignore_index=True)


#################################################################
# Metrics
#################################################################


def generalized_string_metric(x, y, metric):
    """Compute a string metric between x and y.

    x and y can be: two strings (directly computes the metric)
                    two lists of strings (computes the metric between each pair of strings)
                    a string and a list of strings (computes the metric between the string and each string in the list)
                    a list of strings and a list of lists of strings (computes the metric between each string in the first list and each list in the second list)
    """
    # case 1: x and y are strings
    if isinstance(x, str) and isinstance(y, str):
        return metric(x, y)
    # case 2: x is a string and y is a list
    elif isinstance(x, str) and isinstance(y, list):
        return np.array([metric(x, y_i) for y_i in y])
    # case 3: x is a list and y is a string
    elif isinstance(x, list) and isinstance(y, str):
        return np.array([metric(x_i, y) for x_i in x])
    # case 4: two lists
    elif isinstance(x, list) and isinstance(y, list):
        return np.array(
            [generalized_string_metric(x_i, y_i, metric) for x_i, y_i in zip(x, y)]
        )
    raise ValueError(
        "Invalid input types for generalized_string_metric: {}, {}".format(
            type(x), type(y)
        )
    )


def levenshtein_distances(x, y):
    """The Levenstein distances between x and y as a generalized string metric."""
    return generalized_string_metric(x, y, jellyfish.levenshtein_distance)


def strings_equal(x, y):
    """String equality as a generalized string metric."""
    return generalized_string_metric(x, y, lambda x, y: x == y)


def strings_unequal(x, y):
    """String equality as a generalized string metric."""
    return generalized_string_metric(x, y, lambda x, y: x != y)


def metric_with_confidence_interval(y_true, y_pred, metric, confidence_level):
    """Compute a metric. Bootstrap a confidence interval. Compatible with metrics from sklearn.metrics."""
    score = metric(y_true, y_pred)
    res = bootstrap(
        (np.array(y_true), np.array(y_pred)),
        metric,
        vectorized=False,
        paired=True,
        confidence_level=confidence_level,
    )
    return score, res.confidence_interval


def accuracy(y_true, y_pred, confidence_level=0.95):
    """Accuracy with confidence interval."""
    return metric_with_confidence_interval(
        y_true, y_pred, metrics.accuracy_score, confidence_level
    )


def mean_squared_error(y_true, y_pred, confidence_level=0.95):
    """Mean squared error with confidence interval."""
    return metric_with_confidence_interval(
        y_true, y_pred, metrics.mean_squared_error, confidence_level
    )


#################################################################
# Misc.
#################################################################


def adjust_num_prefix_features(original_csv_file, num_prefix_features, new_csv_file):
    """Adjust the number of prefix features proportionally with the number of features in a csv file"""
    original_fn = get_feature_names(original_csv_file)
    new_fn = get_feature_names(new_csv_file)
    adjusted_pf = int(num_prefix_features * len(new_fn) / len(original_fn))
    adjusted_pf = max(adjusted_pf, 1)
    adjusted_pf = min(adjusted_pf, len(new_fn) - 1)
    return adjusted_pf


def get_prefix_features(csv_file, p: float):
    """Returns the names of the first p percent of features in a csv file.

    IMPORTANT: this function always returns a true prefix, i.e. at most 1 and at least len(feature_names) - 1 features.
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    feature_names = get_feature_names(csv_file)
    # error if there is only one feature
    if len(feature_names) == 1:
        raise ValueError(
            "There is only one feature in the csv file. Can't build a valid prefix."
        )
    num_prefix_features = int(p * len(feature_names))
    num_prefix_features = max(num_prefix_features, 1)
    num_prefix_features = min(num_prefix_features, len(feature_names) - 1)
    return feature_names[:num_prefix_features]


def find_nth(s, substring, n):
    """Returns the index of the n-th occurrence of a substring in a string"""
    start = s.find(substring)
    while start >= 0 and n >= 1:
        start = s.find(substring, start + len(substring))
        n -= 1
    return start


def strip_strings_in_dataframe(df: pd.DataFrame):
    """Strip all strings in a dataframe"""
    for column in df.columns:
        if df[column].dtype == object:
            try:
                df[column] = df[column].str.strip()
            except AttributeError:
                pass  # Some columns are objects but not strings (which means .str throws an exception). We can ignore these columns.
    return df


#################################################################
# Printing and html output
#################################################################


# color codes to print with color in the console (from https://gist.github.com/vratiu/9780109)
class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Regular Colors
    Black = "\033[0;30m"  # Black
    Red = "\033[0;31m"  # Red
    Green = "\033[0;32m"  # Green
    Yellow = "\033[0;33m"  # Yellow
    Blue = "\033[0;34m"  # Blue
    Purple = "\033[0;35m"  # Purple
    Cyan = "\033[0;36m"  # Cyan
    White = "\033[0;37m"  # White

    # Background
    On_Black = "\033[40m"  # Black
    On_Red = "\033[41m"  # Red
    On_Green = "\033[42m"  # Green
    On_Yellow = "\033[43m"  # Yellow
    On_Blue = "\033[44m"  # Blue
    On_Purple = "\033[45m"  # Purple
    On_Cyan = "\033[46m"  # Cyan
    On_White = "\033[47m"  # White


def levenshtein_cmd(a: str, b: str):
    """Visualization of the Levenshtein distance between a and b, using color codes to be printed in the console."""
    print_string = ""
    for opcode in levenshtein(a, b)[1]:
        op = opcode["type"]
        a_pos = opcode["i"]
        b_pos = opcode["j"]
        if op == "match":
            print_string += bcolors.Green + b[b_pos] + bcolors.ENDC
        elif op == "insertion":
            print_string += bcolors.Red + b[b_pos] + bcolors.ENDC
        elif op == "deletion":
            print_string += bcolors.Purple + a[a_pos] + bcolors.ENDC
        elif op == "substitution":
            print_string += bcolors.Red + b[b_pos] + bcolors.ENDC
    return print_string


def levenshtein_html(a: str, b: str):
    """HTML visualization of the Levenshtein distance between a and b."""
    html_string = ""
    for opcode in levenshtein(a, b)[1]:
        op = opcode["type"]
        a_pos = opcode["i"]
        b_pos = opcode["j"]
        if op == "match":
            html_string += f'<span style="background-color:#7CFC00">{b[b_pos]}</span>'
        elif op == "insertion":
            html_string += f'<span style="background-color:#EE4B2B">{b[b_pos]}</span>'
        elif op == "substitution":
            html_string += f'<span style="background-color:#EE4B2B">{b[b_pos]}</span>'
        elif op == "deletion":
            html_string += f'<span style="background-color:#BF40BF">{a[a_pos]}</span>'
        if op != "deletion" and b[b_pos] == "\n":
            html_string += "<br>"
    return html_string


def levenshtein_html_legend():
    """HTML legend for the levenshtein visualization."""
    return """'<b>Legend:</b> Prompt, <span style="background-color:#7CFC00">Correct</span>, <span style="background-color:#EE4B2B">Incorrect</span>, <span style="background-color:#BF40BF">Missing</span>'"""


#################################################################
# Levenshtein distance with edits
# from  https://gist.github.com/curzona/9435822
#################################################################


def levenshtein(s1, s2, key=hash):
    """Levenshtein distance with edits."""

    # Based on http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    # Generate the cost matrix for the two strings
    def costmatrix(s1, s2, key=hash):
        rows = []

        previous_row = range(len(s2) + 1)
        rows.append(list(previous_row))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (key(c1) != key(c2))
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

            rows.append(previous_row)

        return rows

    # Trace back through the cost matrix to generate the list of edits
    def backtrace(s1, s2, rows, key=hash):
        i, j = len(s1), len(s2)

        edits = []

        while not (i == 0 and j == 0):
            prev_cost = rows[i][j]

            neighbors = []

            if i != 0 and j != 0:
                neighbors.append(rows[i - 1][j - 1])
            if i != 0:
                neighbors.append(rows[i - 1][j])
            if j != 0:
                neighbors.append(rows[i][j - 1])

            min_cost = min(neighbors)

            if min_cost == prev_cost:
                i, j = i - 1, j - 1
                edits.append({"type": "match", "i": i, "j": j})
            elif i != 0 and j != 0 and min_cost == rows[i - 1][j - 1]:
                i, j = i - 1, j - 1
                edits.append({"type": "substitution", "i": i, "j": j})
            elif i != 0 and min_cost == rows[i - 1][j]:
                i, j = i - 1, j
                edits.append({"type": "deletion", "i": i, "j": j})
            elif j != 0 and min_cost == rows[i][j - 1]:
                i, j = i, j - 1
                edits.append({"type": "insertion", "i": i, "j": j})

        edits.reverse()

        return edits

    rows = costmatrix(s1, s2, key)
    edits = backtrace(s1, s2, rows, key)

    return rows[-1][-1], edits
