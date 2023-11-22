from xgboost import XGBRegressor, XGBClassifier

# scikit linear and logistic regression
from sklearn.linear_model import LinearRegression, LogisticRegression

# scikit scaling and pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from scipy.stats import ttest_ind

import numpy as np
import warnings

import tabmemcheck.llm as llm_utils
import tabmemcheck.utils as utils

from sklearn.preprocessing import LabelEncoder

from tenacity import retry, stop_after_attempt

####################################################################################
# testing row independence
####################################################################################


@retry(
    stop=stop_after_attempt(10)
)  # the automated fitting can fail for an unlucky choice of the test rows (I think. at least it can fail with certain probability due to bad label encoding. this is a quick fix)
def statistical_feature_prediction_test(
    csv_file, feature_name, num_prefix_rows=5, confidence_level=0.95, verbose=False
):
    """Train a gradient boosted tree and a linear classifer to predict the value of feature {feature_name} in the n-th row of the csv file,
      using all the features of the previous {num_prefix_rows} rows.

    Returns: True if the null of no overalp is rejected, False otherwise.
    """
    # load the file as a pandas dataframe
    df = utils.load_csv_df(csv_file)
    feature_names = utils.get_feature_names(csv_file)

    # auto-adjust the number of prefix rows bases on the size of the dataset
    # (it is more important to have a big test set, so that we can detect strong effects (row id) on small datasets with significance)
    num_prefix_rows = 5
    if len(df) < 1000:
        num_prefix_rows = 3
    if len(df) < 500:
        num_prefix_rows = 2
    if len(df) < 200:
        num_prefix_rows = 1

    # we need to make a strict separation between train and test rows
    # this means that we exclude the {num_prefix_rows} rows before any test row from the training set
    test_rows = np.random.choice(
        len(df), size=(len(df) // (1 + num_prefix_rows)) // 2, replace=False
    )

    # regression or classification?
    classification = False
    if df[feature_name].dtype == "object":
        classification = True
    elif (
        len(df[feature_name].unique()) < 25
        and len(df[feature_name].unique()) / len(df) < 0.05
    ):
        # if the feature takes only a couple of values, classification
        df[feature_name] = df[feature_name].astype("category").cat.codes
        classification = True

    # convert all numbers to floats
    for fn in feature_names:
        if df[fn].dtype == "int64":
            df[fn] = df[fn].astype(float)

    # convert stings to categorical features
    for fn in feature_names:
        if df[fn].dtype == "object":
            df[fn] = df[fn].astype("category").cat.codes

    # impute all missing values with the mean
    df = df.fillna(df.mean())

    # construct the prediction problem
    X_train, X_test = [], []
    y_train, y_test = [], []
    for i_row in range(num_prefix_rows, len(df)):
        # the value of the feature in the test row
        y_i = df[feature_name].iloc[i_row]
        # all the values of the previous num_prefix_rows rows
        X_i = df.iloc[i_row - num_prefix_rows : i_row].values.flatten()
        # is this row train, test, or excluded?
        if i_row in test_rows:  # test
            X_test.append(X_i)
            y_test.append(y_i)
        else:
            excluded = False
            for dist in range(num_prefix_rows):
                if i_row + dist + 1 in test_rows:  # excluded
                    excluded = True
            if not excluded:  # train
                X_train.append(X_i)
                y_train.append(y_i)
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    # train a gradient boosted tree and logistic/linear regression
    gbtree = XGBRegressor()
    linear_clf = make_pipeline(StandardScaler(), LinearRegression())
    if classification:
        gbtree = XGBClassifier()
        linear_clf = make_pipeline(StandardScaler(), LogisticRegression())
    # ignore convergence warnings etc.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        gbtree.fit(X_train, y_train)
        linear_clf.fit(X_train, y_train)
    # for the test, we choose the classifier with the lower TRAINING error
    # (we can do this without adjusting the confidence level)
    final_model = gbtree
    if linear_clf.score(X_train, y_train) < gbtree.score(X_train, y_train):
        final_model = linear_clf
    # the final predictions
    y_pred = final_model.predict(X_test)

    # evaluation
    if classification:
        # measure the predictive accuracy
        score, ci = utils.accuracy(y_pred, y_test, confidence_level=confidence_level)
        # the best unconditional predictor: always predicting the most common class
        y_pred = np.repeat(np.argmax(np.bincount(y_train)), len(y_test))
        baseline_score, baseline_ci = utils.accuracy(y_pred, y_test)
        if verbose:
            print(f"Accuracy: {score:.3} ({ci.low:.3}, {ci.high:.3})")
            print(
                f"Baseline (most common class): {baseline_score:.3} ({baseline_ci.low:.3}, {baseline_ci.high:.3})"
            )
    else:
        # measure the mean squared error
        score, ci = utils.mean_squared_error(
            y_pred, y_test, confidence_level=confidence_level
        )
        # the mean absolute error of the mean
        baseline_score, baseline_ci = utils.mean_squared_error(
            np.repeat(np.mean(y_train), len(y_test)), y_test
        )
        if verbose:
            print(f"Mean squared error: {score:.3} ({ci.low:.3}, {ci.high:.3})")
            print(
                f"Baseline (mean): {baseline_score:.3} ({baseline_ci.low:.3}, {baseline_ci.high:.3})"
            )

    # is the gbtree significantly better than the baseline?
    if classification:
        if ci.low > baseline_ci.high:
            return True
    else:
        if ci.high < baseline_ci.low:
            return True
    return False


def row_independence_test(csv_file, confidence_level=0.99):
    """Performs a feature prediction test for each feature in the csv file."""
    df = utils.load_csv_df(csv_file)
    feature_names = utils.get_feature_names(csv_file)
    for feature_name in feature_names:
        # testing each feature separatey, we are performing multiple hypothesis testing
        # we apply the bonferoni correction
        if statistical_feature_prediction_test(
            csv_file,
            feature_name,
            confidence_level=1 - (1 - confidence_level) / len(feature_names),
        ):
            print(
                llm_utils.bcolors.Red
                + f"Row independence rejected at alpha={1-confidence_level:.3}. The rows in this csv file might not be independent!"
                + llm_utils.bcolors.ENDC
            )
            return True
    print(
        llm_utils.bcolors.BOLD
        + f"The null of row indepence was not rejected. The rows in this csv file might still be dependent."
        + llm_utils.bcolors.ENDC
    )
    return False
