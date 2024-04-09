####################################################################################
# This file contains different chat completion functions.
#
# The functions in this file generate, format and send prompts, based on
# the provided csv files. They return the raw model responses, and do not
# perform any tests or analysis. Different tests make use
# of the same chat completion functions.
#
# In the end, almost everything is based on prefix_suffix_chat_completion.
####################################################################################

import numpy as np
import pandas as pd

import tabmemcheck.utils as utils

from tabmemcheck.llm import LLM_Interface, send_chat_completion, send_completion


####################################################################################
# Feature values chat completion function. This function is used for sampling,
# conditional sampling, and prediction.
####################################################################################


def feature_values_chat_completion(
    llm: LLM_Interface,
    csv_file: str,
    system_prompt,
    num_queries,
    few_shot=[],  # list or integer
    cond_feature_names=[],
    fs_cond_feature_names=[],  # a list of lists of conditional feature names for each few-shot example
    add_description=True,
    out_file=None,
):
    """Feature chat completion task. This task asks the LLM to complete the feature values of observations in the dataset.

    The prompt format is the following:
        System: <system_prompt>
            |
            | {few_shot} examples from other csv files.
            |
        User: Dataset: <dataset_name>
              Feature Names: Feature 1, Feature 2, ..., Feature n
              Feature Values: Feature 1 = value 1, Feature 2 = value 2, ..., Feature m = value m
              [Target: Feature k]
        Response: Feature m + 1 = value m + 1, ..., Feature n = value n [Feature k = value k]

    This can be modified in the following ways:
        - Remove dataset description and feature names ({add_description} parameter)
        - don't provide any conditional features
        - Don't use the feature names, but only the values.   (TODO ? or maybe remove, latter for formatter class)

    Options:
        - few_shot: use few-shot examples from other csv files (list), or few_shot examples from the same csv file (int)
        - target & fs_targets: if target is not None, then the LLM is asked to complete only the value of the target feature.

    The feature names are ordered in the prompt as they are ordered in the csv file. In the future we might want to relax this.

    TODO test and debug this function
    """
    # TODO assert that all the given feature names are valid (i.e. occur in the dataset, otherwise throw exception)

    dataset_name = utils.get_dataset_name(csv_file)
    conditional_sampling = (
        cond_feature_names is not None and len(cond_feature_names) > 0
    )

    # if the few-shot argument is a list, then csv_file should not be in there
    # the current option is to remove it (TODO issue warning)
    if isinstance(few_shot, list):
        few_shot = [
            x for x in few_shot if not dataset_name in utils.get_dataset_name(x)
        ]

    # if few-shot is an integer, then include few_shot examples from csv_file
    # this is implemented by replacing few_shot and fs_cond_feature_names with the appropriate lists
    if isinstance(few_shot, int):
        few_shot = [csv_file for _ in range(few_shot)]
        fs_cond_feature_names = [cond_feature_names for _ in range(len(few_shot))]

    # issue a warning if conditional_sampling, but no fs_cond_feature_names
    if conditional_sampling and len(few_shot) > 0 and len(fs_cond_feature_names) == 0:
        print(
            llm.bcolors.WARNING
            + "WARNING: feature_chat_completion: Conditional sampling, but no conditional feature names for the few-shot examples provided."
            + llm.bcolors.ENDC
        )

    # prefixes and suffixes for the main dataset
    if conditional_sampling:
        prefixes, samples = utils.load_cond_samples(
            csv_file, cond_feature_names, add_description=add_description
        )
    else:
        prefix, samples = utils.load_samples(csv_file)
        prefixes = [prefix] * len(samples)

    # prefixes and suffixes for the few-shot examples
    few_shot_prefixes_suffixes = []
    for fs_idx, fs_csv_file in enumerate(few_shot):
        if conditional_sampling:
            fs_prefixes, fs_samples = utils.load_cond_samples(
                fs_csv_file,
                fs_cond_feature_names[fs_idx],
                add_description=add_description,
            )
            few_shot_prefixes_suffixes.append((fs_prefixes, fs_samples))
        else:
            fs_prefix, fs_samples = utils.load_samples(fs_csv_file)
            few_shot_prefixes_suffixes.append(
                ([fs_prefix] * len(fs_samples), fs_samples)
            )

    # execute chat queries
    test_prefixes, test_suffixes, responses = prefix_suffix_chat_completion(
        llm,
        prefixes,
        samples,
        system_prompt,
        few_shot=few_shot_prefixes_suffixes,
        num_queries=num_queries,
        out_file=out_file,
    )

    return test_prefixes, test_suffixes, responses


####################################################################################
# The row chat completion task. This task ask the LLM to predict the next row in the
# csv file, given the previous rows. This task is the basis for the row completion
# test, and also for the first token test.
####################################################################################


def row_chat_completion(
    llm,
    csv_file,
    system_prompt,
    num_prefix_rows=10,
    num_queries=100,
    few_shot=7,
    out_file=None,
    print_levenshtein=False,
):
    """Row  chat completion task. This task ask the LLM to predict the next row in the
    csv file, given the previous rows. This task is the basis for the row completion
    test, and also for the first token test. Uses prefix_suffix_chat_completion."""
    # assert that few_shot is an integer
    assert isinstance(few_shot, int), "For row completion, few_shot must be an integer."

    # load the file as a list of strings
    rows = utils.load_csv_rows(csv_file)

    # prepare data
    prefixes = []
    suffixes = []
    for idx in range(len(rows) - num_prefix_rows):
        prefixes.append("\n".join(rows[idx : idx + num_prefix_rows]))
        suffixes.append(rows[idx + num_prefix_rows])

    test_prefixes, test_suffixes, responses = prefix_suffix_chat_completion(
        llm,
        prefixes,
        suffixes,
        system_prompt,
        few_shot=few_shot,
        num_queries=num_queries,
        out_file=out_file,
        print_levenshtein=print_levenshtein,
    )

    return test_prefixes, test_suffixes, responses


def row_completion(
    llm,
    csv_file,
    num_prefix_rows=10,
    num_queries=100,
    out_file=None,  # TODO support out_file
):
    """Plain language model variant of row_chat_completion"""
    # load the file as a list of strings
    rows = utils.load_csv_rows(csv_file)

    # choose num_queries rows to complete
    prefixes = []
    suffixes = []
    responses = []
    for idx in np.random.choice(
        len(rows) - num_prefix_rows, num_queries, replace=False
    ):
        # prepare query
        prefix = "\n".join(rows[idx : idx + num_prefix_rows])
        suffix = rows[idx + num_prefix_rows]

        # send query
        response = send_completion(llm, prefix, max_tokens=1 + len(suffix))

        # keep only the first row in the response
        response = response.strip("\n").split("\n")[0]

        # store prefix, suffix and response
        prefixes.append(prefix)
        suffixes.append(suffix)
        responses.append(response)

    return prefixes, suffixes, responses


####################################################################################
# Basic completion with a list of strings.
####################################################################################


def chat_completion(
    llm: LLM_Interface,
    strings: list[str],
    system_prompt: str = "You are a helpful assistant that complets the user's input.",
    few_shot=5,
    num_queries=10,
    print_levenshtein=False,
    out_file=None,
    rng=None,
):
    """Basic completion with a chat model and a list of strings."""
    # randomly split the strings into prefixes and suffixes, then use prefix_suffix_chat_completion
    if rng is None:
        rng = np.random.default_rng()
    prefixes = []
    suffixes = []
    for s in strings:
        idx = rng.integers(low=int(len(s) / 3), high=int(2 * len(s) / 3))
        prefixes.append(s[:idx])
        suffixes.append(s[idx:])
    return prefix_suffix_chat_completion(
        llm,
        prefixes,
        suffixes,
        system_prompt,
        few_shot,
        num_queries,
        print_levenshtein,
        out_file,
        rng,
    )


####################################################################################
# Almost all of the different tests that we perform
# can be cast in the prompt structue of
# 'prefix-suffix chat completion'.
# This is implemented by the following function.
####################################################################################


def prefix_suffix_chat_completion(
    llm: LLM_Interface,
    prefixes: list[str],
    suffixes: list[str],
    system_prompt: str,
    few_shot=None,
    num_queries=100,
    print_levenshtein=False,
    out_file=None,
    rng=None,
):
    """A basic chat completion function. Takes a list of prefixes and suffixes and a system prompt.
    Sends {num_queries} prompts of the format

    System: <system_prompt>
        User: <prefix>          |
        Assistant: <suffix>     |
        ...                     | {few_shot} times, or one example from each (prefixes, suffixes) pair in a {few_shot} list.
        User: <prefix>          | In the second case, few_shot = [([prefixes], [suffixes]), ..., ([prefixes], [suffixes])]
        Assistant: <suffix>     |
    User: <prefix>
    Assistant: <response> (=  test suffix?)

    The num_queries prefixes and suffixes are randomly selected from the respective lists.
    The function guarantees that the test suffix (as a complete string) is not contained in any of the few-shot prefixes or suffixes.

    Stores the results in a csv file.

    Returns: the test prefixes, test suffixes, and responses
    """
    assert len(prefixes) == len(
        suffixes
    ), "prefixes and suffixes must have the same length"

    # randomly shuffle the prefixes and suffixes
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.permutation(len(prefixes))
    prefixes = [prefixes[i] for i in idx]
    suffixes = [suffixes[i] for i in idx]

    # the number of points to evaluate
    num_points = min(num_queries, len(prefixes))

    test_prefixes = []
    test_suffixes = []
    responses = []
    for i_testpoint in range(num_points):
        # system prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        # few-shot examples?
        if few_shot is not None:
            # if few_shot is an integer, include few_shot examples from the original prefixes and suffixes
            if isinstance(few_shot, int):
                for _ in range(few_shot):
                    idx = None
                    retries = 0
                    # select a random prefix/suffix pair
                    while (
                        idx is None
                        or idx == i_testpoint
                        # assert that the test suffix is not contained in the few-shot prefixes or suffixes
                        or suffixes[i_testpoint] in prefixes[idx]
                        or suffixes[i_testpoint] in suffixes[idx]
                    ):
                        idx = rng.choice(len(prefixes))
                        retries += 1
                        if retries > 100:
                            raise Exception(
                                "Unable to construct a query where the desired output is not contained in the few-shot data.\nDid you provide the test dataset as few-shot example?"
                            )
                    prefix = prefixes[idx]
                    suffix = suffixes[idx]
                    messages.append({"role": "user", "content": prefix})
                    messages.append({"role": "assistant", "content": suffix})
            # if few_shot is a list of (prefixes, suffixes)-tuples, inlude one example from each tuple
            elif isinstance(few_shot, list):
                for fs_prefixes, fs_suffixes in few_shot:
                    fs_prefix, fs_suffix = None, None
                    retries = 0
                    # select a random prefix/suffix pair
                    while (
                        fs_prefix is None
                        # assert that the test suffix is not contained in the few-shot prefixes or suffixes
                        or suffixes[i_testpoint] in fs_prefix
                        or suffixes[i_testpoint] in fs_suffix
                    ):
                        fs_idx = rng.choice(len(fs_prefixes))
                        fs_prefix = fs_prefixes[fs_idx]
                        fs_suffix = fs_suffixes[fs_idx]
                        retries += 1
                        if retries > 100:
                            raise Exception(
                                "Unable to construct a query where the desired output is not contained in the few-shot data.\nDid you provide the test dataset as few-shot example?"
                            )
                    messages.append({"role": "user", "content": fs_prefix})
                    messages.append({"role": "assistant", "content": fs_suffix})

        # test observation
        test_prefix = prefixes[i_testpoint]
        test_suffix = suffixes[i_testpoint]
        messages.append({"role": "user", "content": test_prefix})
        response = send_chat_completion(llm, messages)
        # store prefix, suffix and response
        test_prefixes.append(test_prefix)
        test_suffixes.append(test_suffix)
        responses.append(response)
        # print the levenshtein distance between the true suffix and the response
        if print_levenshtein:
            print(
                "RESPONSE:",
                utils.levenshtein_cmd(test_suffix, response[: len(test_suffix) + 10]),
            )

    # save the results to file
    if out_file is not None:
        results_df = pd.DataFrame(
            {
                "prefix": test_prefixes,
                "suffix": test_suffixes,
                "response": responses,
            }
        )
        results_df.to_csv(
            out_file,
            index=False,
        )

    return test_prefixes, test_suffixes, responses
