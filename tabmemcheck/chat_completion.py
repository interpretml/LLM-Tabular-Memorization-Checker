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

from typing import Union, Tuple

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
    rng=None,
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
        rng=rng,
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
    rng=None,
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
        rng=rng,
    )

    return test_prefixes, test_suffixes, responses


def row_completion(
    llm,
    csv_file,
    num_prefix_rows=10,
    num_queries=100,
    out_file=None,  # TODO support out_file
    print_levenshtein=False,
    rng=None,
):
    """Plain language model variant of row_chat_completion"""
    # load the file as a list of strings
    rows = utils.load_csv_rows(csv_file)

    # choose num_queries rows to complete
    prefixes = []
    suffixes = []
    responses = []

    if rng is None:
        rng = np.random.default_rng()

    for idx in rng.choice(
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

        # print the levenshtein distance between the true suffix and the response
        if print_levenshtein:
            print(
                utils.levenshtein_cmd(suffix, response[: len(suffix) + 10]),
            )

    return prefixes, suffixes, responses


####################################################################################
# General-purpose chat completion
####################################################################################


def __split_prefix_suffix(
    text: str,
    prefix_length: Union[int, Tuple[int, int]],
    prefix_offset: Union[int, str],
    suffix_length: Union[int, Tuple[int, int]],
    rng,
):
    """Split a string into a prefix and a suffix."""
    # if prefix_length is a tuple, choose a random integer from the given range
    if isinstance(prefix_length, tuple):
        prefix_length = rng.integers(
            low=prefix_length[0], high=prefix_length[1], size=1
        )[0]
    # same for suffix_length
    if isinstance(suffix_length, tuple):
        suffix_length = rng.integers(
            low=suffix_length[0], high=suffix_length[1], size=1
        )[0]
    if prefix_offset == "random":
        prefix_offset = rng.integers(
            low=0, high=len(text) - prefix_length - suffix_length, size=1
        )[0]
    return (
        text[prefix_offset : prefix_offset + prefix_length],
        text[
            prefix_offset
            + prefix_length : prefix_offset
            + prefix_length
            + suffix_length
        ],
    )


def __build_contiguous_query(
    text: str,
    prefix_length: int,  # TODO random prefix and suffix length
    suffix_length: int,
    position: Union[int, str],
    few_shot: int,
    rng,
):
    query_length = (prefix_length + suffix_length) * (1 + few_shot)
    # the length of the string must be at least (prefix_length + suffix_length) * (1 + few_shot)
    assert (
        len(text) >= query_length
    ), "The provided string is too short for the specified prefix and suffix lengths."
    # choose a random sub-string of length query_length
    if position == "random":
        position = rng.integers(low=0, high=len(text) - query_length)
    s_query = text[position : position + query_length]
    # construct few-shot examples
    few_shot_examples = []
    for i_fs in range(few_shot):
        offset = (prefix_length + suffix_length) * i_fs
        few_shot_examples.append(
            (
                [s_query[offset : offset + prefix_length]],
                [
                    s_query[
                        offset + prefix_length : offset + prefix_length + suffix_length
                    ]
                ],
            )
        )
    # prefix and suffix
    prefix = s_query[
        query_length - prefix_length - suffix_length : query_length - suffix_length
    ]
    suffix = s_query[query_length - suffix_length :]
    return few_shot_examples, prefix, suffix


def chat_completion(
    llm: LLM_Interface,
    text: str,
    system_prompt: str = "You are a helpful assistant.",
    prefix_length: int = 300,
    suffix_length: int = 300,
    position: Union[int, str] = 0,
    few_shot=5,  # integer, or list [str, ..., str]
    contiguous=False,
    num_queries=1,
    print_levenshtein=False,
    out_file=None,
    rng=None,
):
    """Prompt a chat model to (verbatim) complete a text.

    Args:
        llm (LLM_Interface): The LLM.
        text (str): The text that we ask the model to complete.
        system_prompt (str, optional): _description_. Defaults to "You are a helpful assistant.".
        prefix_length (int, optional): _description_. Defaults to 300.
        suffix_length (int, optional): _description_. Defaults to 300.
        position (Union[int, str], optional): _description_. Defaults to 0.
        few_shot (int, optional): _description_. Defaults to 5.
        num_queries (int, optional): _description_. Defaults to 1.
        print_levenshtein (bool, optional): _description_. Defaults to False.
        out_file (_type_, optional): _description_. Defaults to None.
        rng (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if rng is None:
        rng = np.random.default_rng()

    if contiguous:
        # few-shot has to be an integer
        assert isinstance(
            few_shot, int
        ), "For contiguous chat completion, few_shot must be an integer."
        # both prefix_length and suffix_length have to be specified
        assert (
            prefix_length is not None and suffix_length is not None
        ), "For contiguous chat completion, both prefix_length and suffix_length have to be specified."
        prefixes, suffixes, responses = [], [], []
        for _ in range(num_queries):
            # select a random string and build the query
            few_shot_examples, prefix, suffix = __build_contiguous_query(
                text, prefix_length, suffix_length, position, few_shot, rng
            )
            # send query
            prefix, suffix, response = prefix_suffix_chat_completion(
                llm,
                [prefix],
                [suffix],
                system_prompt,
                few_shot=few_shot_examples,
                num_queries=1,
                print_levenshtein=print_levenshtein,
                out_file=out_file,
                rng=rng,
            )
            prefixes.append(prefix)
            suffixes.append(suffix)
            responses.append(response)
        return prefixes, suffixes, responses

    # non-contiguous
    prefixes, suffixes, responses = [], [], []
    for _ in range(num_queries):
        # test prefix and suffix
        prefix, suffix = __split_prefix_suffix(
            text, prefix_length, position, suffix_length, rng
        )
        # few shot examples
        if isinstance(few_shot, list):  # few_shot is list of strings
            few_shot_examples = [
                __split_prefix_suffix(s, prefix_length, "random", suffix_length, rng)
                for s in few_shot
            ]
            few_shot_examples = [([fs[0]], [fs[1]]) for fs in few_shot_examples]
        else:  # few_shot is integer
            # remove the selected prefix and suffix from the text
            remaining_text = text.replace(prefix, "")
            remaining_text = remaining_text.replace(suffix, "")
            # construct few-shot examples from the remaining text. not perfect but it works.
            few_shot_examples = [
                __split_prefix_suffix(
                    remaining_text, prefix_length, "random", suffix_length, rng
                )
                for _ in range(few_shot)
            ]
            few_shot_examples = [([fs[0]], [fs[1]]) for fs in few_shot_examples]
        # send query
        prefix, suffix, response = prefix_suffix_chat_completion(
            llm,
            [prefix],
            [suffix],
            system_prompt,
            few_shot_examples,
            num_queries=1,
            print_levenshtein=print_levenshtein,
            out_file=out_file,
            rng=rng,
        )
        prefixes.append(prefix)
        suffixes.append(suffix)
        responses.append(response)
    return prefixes, suffixes, responses


####################################################################################
# Many tests can be cast in the prompt structue of 'prefix-suffix chat completion'.
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
    """A general-purpose chat completion function. Given prefixes, suffixes, and few-shot examples, this function sends {num_queries} LLM queries of the format

    System: <system_prompt>
        User: <prefix>          |
        Assistant: <suffix>     |
        ...                     | {few_shot} times, or one example from each (prefixes, suffixes) pair in a {few_shot} list.
        User: <prefix>          | In the second case, few_shot = [([prefixes], [suffixes]), ..., ([prefixes], [suffixes])]
        Assistant: <suffix>     |
    User: <prefix>
    Assistant: <response> (=  test suffix?)

    The prefixes, suffixes are and few-shot examples are randomly selected.
    
    This function guarantees that the test suffix (as a complete string) is not contained in any of the few-shot prefixes or suffixes (a useful sanity check, we don't want to provide the desired response anywhere in the context).

    Args:
        llm (LLM_Interface): The LLM.
        prefixes (list[str]): A list of prefixes.
        suffixes (list[str]): A list of suffixes.
        system_prompt (str): The system prompt.
        few_shot (_type_, optional): Either an integer, to select the given number of few-shot examples from the list of prefixes and suffixes. Or a list [([prefixes], [suffixes]), ..., ([prefixes], [suffixes])] to select one few-shot example from each list. Defaults to None.
        num_queries (int, optional): The number of queries. Defaults to 100.
        print_levenshtein (bool, optional): Visualize the Levenshtein string distance between test suffixes and LLM responses. Defaults to False.
        out_file (_type_, optional): Save all queries to a CSV file. Defaults to None.
        rng (_type_, optional): _description_. Defaults to None.

    Raises:
        Exception: It an error occurs.

    Returns:
        tuple: A tuple of test prefixes, test suffixes, and responses.
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
