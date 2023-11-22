####################################################################################
# LEGACY CODE THAT STILL NEEDS TO BE REFACTORED
####################################################################################

import os

import numpy as np
import pandas as pd

import tabmemcheck.analysis as analysis
import tabmemcheck.utils as utils
import tabmemcheck.llm as llm_utils

from tabmemcheck.row_independence import statistical_feature_prediction_test

from tabmemcheck.chat_completion import (
    prefix_suffix_chat_completion,
    row_chat_completion,
    feature_values_chat_completion,
)


####################################################################################
# Prediction and sampling
####################################################################################
from collections import Counter


def predict(csv_file, system_prompt, feature_name: str, num_queries=100, few_shot=7):
    """Predict the feature {feature_name}, using all the other features in the csv file.

    Reports the accuracy / mse.

    The basic prompt format is the following:
        System: <system_prompt>
        User: Feature 1 = value 1, Feature 2 = value 2, ..., Feature n = value n
        Response: Target Feature = value

    Inludes {few_shot} examples from the same csv file.

    TODO allow to stratify the few-shot examples by the target feature.
    """
    # all the other features are the conditional features
    feature_names = utils.get_feature_names(csv_file)
    cond_feature_names = [f for f in feature_names if f != feature_name]

    # run the test
    test_prefixes, test_suffixes, responses = feature_values_chat_completion(
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        add_description=False,
    )

    # parse the model responses
    response_df = utils.parse_feature_stings(responses, [feature_name])
    test_suffix_df = utils.parse_feature_stings(test_suffixes, [feature_name])

    # is the classification or regression?
    df = utils.load_csv_df(csv_file)
    is_classification = False
    if df[feature_name].dtype == "object":
        is_classification = True

    # compute the accuracy/mse
    y_true = test_suffix_df[feature_name]
    y_pred = response_df[feature_name]

    if is_classification:
        score, ci = utils.accuracy(y_true, y_pred)
        print(
            llm_utils.bcolors.BOLD
            + "Accuracy: "
            + llm_utils.bcolors.ENDC
            + f"{score:.3} ({ci.low:.3}, {ci.high:.3})"
        )
        # TODO replace test with train here
        baseline_score, baseline_ci = utils.accuracy(
            y_true, np.repeat(Counter(y_true).most_common(1)[0][0], len(y_true))
        )
        print(
            llm_utils.bcolors.BOLD
            + "Baseline (most common class): "
            + llm_utils.bcolors.ENDC
            + f"{baseline_score:.3} ({baseline_ci.low:.3}, {baseline_ci.high:.3})"
        )
    else:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        score, ci = utils.mse(y_true, y_pred)
        print(
            llm_utils.bcolors.BOLD
            + "Mean-squared-error: "
            + llm_utils.bcolors.ENDC
            + f"{score:.3} ({ci.low:.3}, {ci.high:.3})"
        )
        baseline_score, baseline_ci = utils.mse(
            y_true, np.repeat(np.mean(y_true), len(y_true))
        )
        print(
            llm_utils.bcolors.BOLD
            + "Baseline (mean): "
            + llm_utils.bcolors.ENDC
            + f"{baseline_score:.3} ({baseline_ci.low:.3}, {baseline_ci.high:.3})"
        )


####################################################################################
# Mode test and conditional completion test for learning (does the model know the most
# important statistics of the data distribution?)
####################################################################################


# TODO
def mode_test():
    pass


def conditional_completion_test(
    csv_file: str,
    system_prompt,
    feature_name: str,
    num_queries=250,
    prefix_length=[0, 0.25, 0.5, 0.75, 1],
    few_shot=[],
    out_file=None,
):
    """Conditional completion test for conditional distribution modelling.

    The task is to always predict the feature {feature_name}, give different conditional features.

    The prompt format is the following:
        System: <system_prompt>
            |
            | {few_shot} examples from other csv files.
            |
        User: Dataset: <dataset_name>
              Feature Names: Feature 1, Feature 2, ..., Feature m, Feature {feature_name}, Feature m + 2, ..., Feature n
              Feature Values: Feature 1 = value 1, Feature 2 = value 2, ..., Feature m = value m
        Response: {feature_name} = value m + 1 [, Feature {feature_name}, Feature m + 2, ..., Feature n]

    We ask the model to provide {num_queries} completions for each prefix length in {prefix_length}.

    The ordering of the feature names will be the same as in the csv file, except for the feature {feature_name}.

    The test computes the p-value of the hypothesis that the completions are unconditional.
    """
    # all the other features are the conditional features
    feature_names = utils.get_feature_names(csv_file)
    cond_feature_names = [f for f in feature_names if f != feature_name]

    # in this task, the order of the features is not the same as in the original csv file.
    # therefore, we shuffle the order of the features in the few-shot csv files.
    shuffled_few_shot = []
    for fs_csv_file in few_shot:
        df = utils.load_csv_df(fs_csv_file)
        df = df.sample(frac=1, axis=1)
        shuffled_few_shot.append(
            utils.tmp_csv_file(df, utils.get_dataset_name(fs_csv_file))
        )
    few_shot = shuffled_few_shot

    # feature names in the few-shot files
    fs_fns = [utils.get_feature_names(fs_csv_file) for fs_csv_file in few_shot]

    # fs_target_fns = [np.random.choice(fns) for fns in fs_fns]
    # fs_cond_fns = [
    #    [f for f in fns if f != fs_target_fn]
    #    for fns, fs_target_fn in zip(fs_fns, fs_target_fns)
    # ]

    # estimate and set the maximum number of tokens to speed up the experiment
    df = utils.load_csv_df(csv_file)
    max_tokens = (
        df[feature_name]
        .astype(str)
        .apply(lambda x: llm_utils.num_tokens_from_string(f"{feature_name} = " + x))
        .astype(int)
        .max()
    )
    llm_utils.llm_max_tokens = int(1.1 * max_tokens) + 3

    # create a data frame to hold the model responses
    df = pd.DataFrame(columns=["num_prefix_features"].extend(feature_names))
    # one completion task for each prefix length
    for p in prefix_length:
        # the conditional features that we use
        p_cond_fns = cond_feature_names[: int(p * len(cond_feature_names))]
        p_fs_cond_fns = [fns[: min(int(p * len(fns)), len(fns) - 1)] for fns in fs_fns]

        # we create a temporary version of the dataset where the features are ordered as {conditional features}, {target feature}, {other features}
        # this is the order in which we will ask the model to predict the target feature
        p_df = utils.load_csv_df(csv_file)
        p_df = p_df[
            [
                *p_cond_fns,
                feature_name,
                *[f for f in cond_feature_names if f not in p_cond_fns],
            ]
        ]
        p_csv_file = CSVFile.from_df(p_df, utils.get_dataset_name(csv_file)).csv_file

        # run the task
        test_prefixes, test_suffixes, responses = feature_values_chat_completion(
            p_csv_file,
            system_prompt,
            num_queries,
            few_shot,
            p_cond_fns,
            p_fs_cond_fns,
            add_description=True,
            out_file=None,
        )

        # parse the prefixes and response
        test_prefixes_df = utils.parse_feature_stings(test_prefixes, feature_names)
        response_df = utils.parse_feature_stings(responses, feature_names)

        # drop all columns other than feature_name from the response_df
        response_df = response_df[[feature_name]]

        # add all the columns in the response_df to the test_suffix_df
        test_prefixes_df = pd.concat([test_prefixes_df, response_df], axis=1)

        # add empty columns for the features that are not in the prefix
        for f in feature_names:
            if f not in test_prefixes_df.columns:
                test_prefixes_df[f] = [""] * test_prefixes_df.shape[0]

        # add the number of prefix features as a column
        test_prefixes_df["num_prefix_features"] = [
            len(p_cond_fns)
        ] * test_prefixes_df.shape[0]

        # store results
        df = pd.concat(
            [
                df,
                test_prefixes_df,
            ],
            ignore_index=True,
        )

    # save the dataframe
    if out_file is not None:
        df.to_csv(out_file, index=False)

    # analysis, given the dataframe with the results
    return analysis.conditional_completion_test(df)


def ordered_completion(
    csv_file,
    system_prompt,
    feature_name: str,
    num_queries=250,
    few_shot=[],
    out_file=None,
):
    """Ordered, conditional completion of feature {feature_name}.

    The prompt format is the following:
        # TODO
    """
    feature_names = utils.get_feature_names(csv_file)
    fs_feature_names = [
        utils.get_feature_names(fs_csv_file) for fs_csv_file in few_shot
    ]

    # the index of the completion feature
    feature_idx = feature_names.index(feature_name)

    # the relative number of feature names that we use as conditional features
    p = feature_idx / len(feature_names)
    p = max(0.001, min(0.999, p))

    # TODO handle edge case no conditional features
    cond_feature_names = utils.get_prefix_features(csv_file, p)
    fs_cond_feature_names = [
        utils.get_prefix_features(fs_csv_file, p) for fs_csv_file in few_shot
    ]

    # estimate and set the maximum number of tokens to speed up the experiment
    df = utils.load_csv_df(csv_file)
    max_tokens = (
        df[feature_name]
        .astype(str)
        .apply(lambda x: llm_utils.num_tokens_from_string(f"{feature_name} = " + x))
        .astype(int)
        .max()
    )
    llm_utils.llm_max_tokens = int(1.1 * max_tokens) + 3
    print(f"Setting max_tokens to {llm_utils.llm_max_tokens}")

    # log model queries
    dataset_name = utils.get_dataset_name(csv_file)
    exp_name = f"{dataset_name}-ordered-completion-{feature_name}"
    llm_utils.set_logging_task(exp_name)

    # run the task
    test_prefixes, test_suffixes, responses = feature_values_chat_completion(
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        fs_cond_feature_names,
        add_description=True,
        out_file=out_file,
    )
