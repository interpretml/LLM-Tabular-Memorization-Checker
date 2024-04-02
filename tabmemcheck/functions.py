import os
from typing import Any, Union

import numpy as np
import pandas as pd

from difflib import SequenceMatcher

import tabmemcheck as tabmem

import tabmemcheck.analysis as analysis
import tabmemcheck.utils as utils

from tabmemcheck.llm import (
    LLM_Interface,
    ChatWrappedLLM,
    send_chat_completion,
    send_completion,
    bcolors,
)

from tabmemcheck.row_independence import statistical_feature_prediction_test

from tabmemcheck.chat_completion import (
    prefix_suffix_chat_completion,
    row_chat_completion,
    row_completion,
    feature_values_chat_completion,
)


DEFAULT_FEW_SHOT_CSV_FILES = [
    "iris.csv",
    "adult-train.csv",
    "openml-diabetes.csv",
    "uci-wine.csv",
    "california-housing.csv",
]


def __difflib_similar(csv_file_1, csv_file_2):
    sm = SequenceMatcher(
        None, utils.load_csv_string(csv_file_1), utils.load_csv_string(csv_file_2)
    )
    if sm.quick_ratio() > 0.9:
        return sm.ratio() > 0.9
    return False


def __validate_few_shot_files(csv_file, few_shot_csv_files):
    """check if the csv_file is contained in the few_shot_csv_files."""
    dataset_name = utils.get_dataset_name(csv_file)
    few_shot_names = [utils.get_dataset_name(x) for x in few_shot_csv_files]
    if dataset_name in few_shot_names:
        # replace the dataset_name with open-ml diabetes
        few_shot_csv_files = [
            x for x in few_shot_csv_files if utils.get_dataset_name(x) != dataset_name
        ]
        few_shot_csv_files.append("openml-diabetes.csv")
    # now test with difflib if the dataset contents are very similar
    for fs_file in few_shot_csv_files:
        if __difflib_similar(csv_file, fs_file):
            print(
                bcolors.BOLD
                + "Warning: "
                + bcolors.ENDC
                + f"The dataset is very similar to the few-shot dataset {utils.get_dataset_name(fs_file)}."
            )
    return few_shot_csv_files


def __llm_setup(llm: Union[LLM_Interface, str]):
    # if llm is a string, assume open ai model
    if isinstance(llm, str):
        llm = tabmem.openai_setup(llm)
    return llm


def __print_info(csv_file, llm, few_shot_csv_files):
    """Print some information about the csv file and the model."""
    print(
        bcolors.BOLD
        + "Dataset: "
        + bcolors.ENDC
        + f"{utils.get_dataset_name(csv_file)}"
    )
    print(bcolors.BOLD + "Model: " + bcolors.ENDC + f"{llm}")
    print(
        bcolors.BOLD
        + "Few-Shot: "
        + bcolors.ENDC
        + ", ".join(
            [utils.get_dataset_name(fs_csv_file) for fs_csv_file in few_shot_csv_files]
        )
    )


####################################################################################
# All the tests
####################################################################################


def run_all_tests(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    few_shot_csv_files=DEFAULT_FEW_SHOT_CSV_FILES,
    feature_name=None,
):
    llm = __llm_setup(llm)
    few_shot_csv_files = __validate_few_shot_files(csv_file, few_shot_csv_files)
    __print_info(csv_file, llm, few_shot_csv_files)

    feature_names_test(csv_file, llm, few_shot_csv_files=few_shot_csv_files)

    # todo feature values

    header_test(csv_file, llm, few_shot_csv_files=few_shot_csv_files)

    # draw 10 zero-knowledge samples
    print(
        bcolors.BOLD
        + "Drawing 10 zero-knowledge samples at temperature 0.7:"
        + bcolors.ENDC
    )
    temp = tabmem.config.temperature
    tabmem.config.temperature = 0.7
    samples_df = sample(
        csv_file, llm, num_queries=10, few_shot_csv_files=few_shot_csv_files
    )
    # print the data frame unless it is empty
    if (not samples_df.empty) and len(samples_df) > 0:
        pd.set_option("display.expand_frame_repr", False)
        print(samples_df)
        if len(samples_df) < 10:
            print(f"The model provided {len(samples_df)} valid samples.")
    else:
        print("The model was not able to provide valid samples.")
    tabmem.config.temperature = temp

    row_completion_test(csv_file, llm, num_queries=25)
    feature_completion_test(csv_file, llm, num_queries=25, feature_name=feature_name)
    first_token_test(csv_file, llm, num_queries=25)


####################################################################################
# Feature Names
####################################################################################


def feature_names_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    num_prefix_features: int = None,
    few_shot_csv_files=DEFAULT_FEW_SHOT_CSV_FILES,
    system_prompt: str = "default",
):
    """Test if the model knows the names of the features.

    The prompt format is:
        System: <system_prompt>
        User: Dataset: <dataset_name>
              Feature 1, Feature 2, ..., Feature n
        Response: Feature n+1, Feature n+2, ..., Feature m

    This can be modified in the following ways:
    - Include few-shot examples from other csv files.
    """
    llm = __llm_setup(llm)
    few_shot_csv_files = __validate_few_shot_files(csv_file, few_shot_csv_files)

    # default system prompt?
    if system_prompt == "default":
        system_prompt = tabmem.config.system_prompts["feature-names"]

    dataset_name = utils.get_dataset_name(csv_file)
    feature_names = utils.get_feature_names(csv_file)

    # by default, use 1/4 of the features as prefix, but at least one
    if num_prefix_features is None:
        num_prefix_features = max(1, len(feature_names) // 4)

    # remove the current csv file from the few-shot csv files should it be present there
    few_shot_csv_files = [x for x in few_shot_csv_files if not dataset_name in x]

    # setup for the few-shot examples
    fs_dataset_names = [utils.get_dataset_name(x) for x in few_shot_csv_files]
    fs_feature_names = [
        utils.get_feature_names(fs_csv_file) for fs_csv_file in few_shot_csv_files
    ]
    fs_prefix_feature = [
        utils.adjust_num_prefix_features(csv_file, num_prefix_features, fs_csv_file)
        for fs_csv_file in few_shot_csv_files
    ]

    if llm.chat_mode:
        # construt the prompt
        prefixes = [
            f"Dataset: {dataset_name}. Feature Names: "
            + ", ".join(feature_names[:num_prefix_features])
        ]
        suffixes = [", ".join(feature_names[num_prefix_features:])]

        few_shot = []
        for fs_dataset_name, fs_feature_name, fs_prefix_feature in zip(
            fs_dataset_names, fs_feature_names, fs_prefix_feature
        ):
            few_shot.append(
                (
                    [
                        f"Dataset: {fs_dataset_name}. Feature Names: "
                        + ", ".join(fs_feature_name[:fs_prefix_feature])
                    ],
                    [", ".join(fs_feature_name[fs_prefix_feature:])],
                )
            )

        # execute the the prompt
        _, _, responses = prefix_suffix_chat_completion(
            llm,
            prefixes,
            suffixes,
            system_prompt,
            few_shot=few_shot,
            num_queries=1,
        )
        response = responses[0]
    else:
        # construct the prompt
        prompt = ""
        for fs_dataset_name, fs_feature_name, fs_prefix_feature in zip(
            fs_dataset_names, fs_feature_names, fs_prefix_feature
        ):
            prompt += (
                f"Dataset: {fs_dataset_name}.\nNumber of Features: {len(fs_feature_name)}\nFeature Names: "
                + ", ".join(fs_feature_name)
                + "\n\n"
            )
        prompt += (
            f"Dataset: {dataset_name}\nNumber of Features: {len(feature_names)}\nFeature Names: "
            + ", ".join(feature_names[:num_prefix_features])
            + ", "
        )

        # execute the prompt
        response = send_completion(llm, prompt)

        # consider the response only until the first '\n\n'
        idx = response.find("\n\n")
        if idx != -1:
            response = response[:idx]

    print(
        bcolors.BOLD
        + "Feature Names Test\nFeature Names:    "
        + bcolors.ENDC
        + ", ".join(feature_names[num_prefix_features:])
        + bcolors.BOLD
        + "\nModel Generation: "
        + bcolors.ENDC
        + response
    )

    # TODO do some sort of evaluation
    # for example, return true if it completes all but X of the feature names, correcting for upper/lower case
    # at least do formatted printing of the results


####################################################################################
# Feature Values
####################################################################################


####################################################################################
# Header Test
####################################################################################


def header_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    split_rows: list[int] = [2, 4, 6, 8],
    completion_length: int = 500,
    few_shot_csv_files: list[str] = DEFAULT_FEW_SHOT_CSV_FILES,
    system_prompt: str = "default",
):
    """Header test, using other csv files as few-shot examples.

    Splits the csv file at random positions in rows 2, 4, 6, and 8. Performs 1 query for each split. Reports the best completion.

    NOTE: This test might fail if the header and rows of the csv file are very long, and the model has a small context window.
    NOTE: in the end, this is the case for all of our tests :)
    """
    llm = __llm_setup(llm)
    few_shot_csv_files = __validate_few_shot_files(csv_file, few_shot_csv_files)

    # default system prompt?
    if system_prompt == "default":
        system_prompt = tabmem.config.system_prompts["header"]

    # load the csv file as a single contiguous string. also load the rows to determine offsets within the string
    data = utils.load_csv_string(csv_file, header=True)
    csv_rows = utils.load_csv_rows(csv_file, header=True)

    # load the few-shot examples
    few_shot_data = []
    for fs_csv_file in few_shot_csv_files:
        fs_data = utils.load_csv_string(fs_csv_file, header=True)
        few_shot_data.append(fs_data)

    # perform the test multiple times, cutting the dataset at random positions in rows split_rows
    num_completions = -1
    header, completion = None, None
    for i_row in split_rows:
        offset = np.sum([len(row) for row in csv_rows[: i_row - 1]])
        offset += np.random.randint(
            len(csv_rows[i_row]) // 3, 2 * len(csv_rows[i_row]) // 3
        )
        prefixes = [data[:offset]]
        suffixes = [data[offset : offset + completion_length]]
        few_shot = [
            ([fs_data[:offset]], [fs_data[offset : offset + completion_length]])
            for fs_data in few_shot_data
        ]

        # chat mode: use few-shot examples
        if llm.chat_mode:
            _, _, response = prefix_suffix_chat_completion(
                llm, prefixes, suffixes, system_prompt, few_shot=few_shot, num_queries=1
            )
            response = response[0]
        else:  # otherwise, plain completion
            response = send_completion(llm, prefixes[0])

        # find the first digit where the response and the completion disagree
        idx = -1000
        for idx, (c, r) in enumerate(zip(data[offset:], response)):
            if c != r:
                break
        if idx == len(response) - 1 and response[idx] == data[offset + idx]:
            idx += 1  # no disagreement found, set idx to length of the response

        # is this the best completion so far?
        if idx > num_completions:
            num_completions = idx
            header = prefixes[0]
            completion = response

    # for the printing, we first color all green up to the first disagreement
    completion_print = bcolors.Green + completion[:num_completions]

    # then color red up to the beginning of the next row, if any
    remaining_completion = completion[num_completions:]
    idx = remaining_completion.find("\n")
    if idx == -1:
        completion_print += bcolors.Red + remaining_completion
    else:
        completion_print += bcolors.Red + remaining_completion[:idx] + "\n"
        remaining_completion = remaining_completion[idx + 1 :]

        # for all additional rows, green up to the first disagreement, all red after that
        completion_rows = remaining_completion.split("\n")

        # the corresponding next row in the csv file
        data_idx = data[len(header) + num_completions :].find("\n")
        data_rows = data[len(header) + num_completions + data_idx + 1 :].split("\n")

        for completion_row, data_row in zip(completion_rows, data_rows):
            if completion_row == data_row:
                completion_print += bcolors.Green + completion_row + "\n"
                continue
            # not equal, find the first disagreement
            idx = -1000
            for idx, (c, r) in enumerate(zip(data_row, completion_row)):
                if c != r:
                    break
            if idx == len(completion_row) - 1 and completion_row[idx] == data_row[idx]:
                idx += 1
            # print first part green, second part red
            completion_print += (
                bcolors.Green
                + completion_row[:idx]
                + bcolors.Red
                + completion_row[idx:]
                + "\n"
            )

    # remove final new line
    completion_print = completion_print.rstrip("\n")

    # print the result
    print(
        bcolors.BOLD
        + "Header Test: "
        + bcolors.ENDC
        + bcolors.Black
        + header
        + completion_print
        + bcolors.ENDC
        + bcolors.BOLD
        + "\nHeader Test Legend:  "
        + bcolors.ENDC
        + "Prompt "
        + bcolors.Green
        + "Correct "
        + bcolors.Red
        + "Incorrect"
        + bcolors.ENDC
    )

    # TODO return true if it completes the given row, as well as the next row.
    # TODO count the number of correctly completed rows and print this number


####################################################################################
# Row Completion
####################################################################################


def row_completion_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    num_prefix_rows=10,
    num_queries=50,
    few_shot=7,
    out_file=None,
    system_prompt: str = "default",
):
    """Row completion test: Complete the next row of the csv file, given the previous rows."""
    llm = __llm_setup(llm)

    if system_prompt == "default":  # default system prompt?
        system_prompt = tabmem.config.system_prompts["row-completion"]

    # what fraction of the rows are duplicates?
    rows = utils.load_csv_rows(csv_file)
    frac_duplicates = 1 - len(set(rows)) / len(rows)
    if frac_duplicates == 0:
        print(
            bcolors.BOLD
            + "Info: "
            + bcolors.ENDC
            + "All the rows in the dataset are unique."
        )
    else:
        print(
            bcolors.BOLD
            + "Info: "
            + bcolors.ENDC
            + f"{100*frac_duplicates:.2f}% of the rows in this dataset are duplicates."
        )

    # ask the model to perform row chat completion (execute the the prompt)
    if llm.chat_mode:
        test_prefixes, test_suffixes, responses = row_chat_completion(
            llm,
            csv_file,
            system_prompt,
            num_prefix_rows,
            num_queries,
            few_shot,
            out_file,
        )
    else:
        test_prefixes, test_suffixes, responses = row_completion(
            llm, csv_file, num_prefix_rows, num_queries, out_file
        )

    # count the number of exact matches
    # NOTE here we assume that the test suffix is a single row that is unique, i.e. no duplicate rows
    num_exact_matches = 0
    for test_suffix, response in zip(test_suffixes, responses):
        if test_suffix.strip() in response.strip():
            num_exact_matches += 1

    # the statistical test using the levenshtein distance TODO taken out of current version although it works
    # test_prefix_rows = [prefix.split("\n") for prefix in test_prefixes]
    # test_result = analysis.levenshtein_distance_t_test(
    #    responses, test_suffixes, test_prefix_rows
    # )

    # print the result
    print(
        bcolors.BOLD
        + "Row Completion Test: "
        + bcolors.ENDC
        + f"{num_exact_matches}/{num_queries} exact matches."
        # + bcolors.BOLD
        # + "\nLevenshtein distance test (p-value): "
        # + bcolors.ENDC
        # + f"{test_result.pvalue:.3f}."
    )

    return test_prefixes, test_suffixes, responses


####################################################################################
# Feature Completion
####################################################################################


def feature_completion_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    feature_name: str = None,
    num_queries=100,
    few_shot=5,
    out_file=None,
    system_prompt: str = "default",
):
    """Feature completion test where we attempt to predict a single rare feature & count the number of exact matches.

    The basic prompt format is the following:
        System: <system_prompt>
        User: Feature 1 = value 1, Feature 2 = value 2, ..., Feature n = value n
        Response: Feature {feature_name} = value

    This can be modified in the following ways:
        - Include few-shot examples from other csv files.
        - Don't use the feature names, but only the values.
    """
    llm = __llm_setup(llm)

    # TODO statistical analysis of the uniqueness of the feature (i.e., is the test appropriate?!)
    if system_prompt == "default":  # default system prompt?
        system_prompt = tabmem.config.system_prompts["feature-completion"]

    # if no feature value is provided, automatically select the most unique feature
    if feature_name is None:
        feature_name, frac_unique_values = analysis.find_most_unique_feature(csv_file)
        print(
            bcolors.BOLD
            + "Info: "
            + bcolors.ENDC
            + f"Using feature {feature_name} with {100*frac_unique_values:.2f}% unique values."
        )

    # all the other features are the conditional features
    feature_names = utils.get_feature_names(csv_file)
    cond_feature_names = [f for f in feature_names if f != feature_name]

    if not llm.chat_mode:  # wrap base model to take chat queries

        def build_prompt(messages):
            prompt = ""
            for m in messages:
                if m["role"] == "user":
                    prompt += m["content"]
                elif m["role"] == "assistant":
                    prompt += ", " + m["content"] + "\n\n"
            prompt += ", "
            return prompt

        llm = ChatWrappedLLM(llm, build_prompt, ends_with="\n\n")

    # execute the prompt
    _, test_suffixes, responses = feature_values_chat_completion(
        llm,
        csv_file,
        system_prompt,
        num_queries,
        few_shot,
        cond_feature_names,
        add_description=False,
        out_file=out_file,
    )

    # parse the model responses
    response_df = utils.parse_feature_stings(responses, [feature_name])
    test_suffix_df = utils.parse_feature_stings(test_suffixes, [feature_name])

    # count number of exact matches
    num_exact_matches = np.sum(
        response_df[feature_name] == test_suffix_df[feature_name]
    )

    # print the result
    print(
        bcolors.BOLD
        + f'Feature Completion Test ("{feature_name}"): '
        + bcolors.ENDC
        + bcolors.Black
        + f"{num_exact_matches}/{num_queries} exact matches."
        + bcolors.ENDC
    )


####################################################################################
# First Token Test
####################################################################################


def first_token_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    num_prefix_rows=10,
    num_queries=100,
    few_shot=7,
    out_file=None,
    system_prompt: str = "default",
):
    """First token test: Complete the first token of the next row of the csv file, given the previous rows."""
    llm = __llm_setup(llm)

    if (
        system_prompt == "default"
    ):  # default system prompt? (the first token test asks the model to complete the same task as row completion, only the evaluation is different)
        system_prompt = tabmem.config.system_prompts["row-completion"]

    # determine the number of digits that the first token should have
    num_digits = analysis.build_first_token(csv_file)

    # run a feature prediction test to see if the first token is actually random
    df = utils.load_csv_df(csv_file)
    rows = utils.load_csv_rows(csv_file, header=False)
    df["FIRST_TOKEN_TEST_ROW"] = [r[:num_digits] for r in rows]
    df["FIRST_TOKEN_TEST_ROW"] = df["FIRST_TOKEN_TEST_ROW"].astype(str)
    tmp_csv_file = utils.tmp_csv_file(
        df, utils.get_dataset_name(csv_file) + ".csv"
    )  # save the df to a tmp csv file
    rejected = statistical_feature_prediction_test(
        tmp_csv_file,
        "FIRST_TOKEN_TEST_ROW",
        num_prefix_rows=5,
        confidence_level=0.99,
    )

    # the most common first token
    most_common_first_token = df["FIRST_TOKEN_TEST_ROW"].value_counts().index[0]
    # print(most_common_first_token)

    # if the feature prediction test rejects randomness, refuse to run the test
    if rejected:
        print(
            bcolors.BOLD
            + "Info: "
            + bcolors.ENDC
            + "Aborting the first token test because the first token does not seem to be random.\nThe most likely reason for this is that the rows in the csv file are not random.\nFor example, the first feature might be the id of the observation."
        )
        return

    #  set max_tokens to the number of digits (speedup)
    prev_max_tokes = tabmem.config.max_tokens
    tabmem.config.max_tokens = num_digits

    # perform a row completion task
    if llm.chat_mode:
        _, test_suffixes, responses = row_chat_completion(
            llm,
            csv_file,
            system_prompt,
            num_prefix_rows,
            num_queries,
            few_shot,
            out_file,
        )
    else:
        _, test_suffixes, responses = row_completion(
            llm, csv_file, num_prefix_rows, num_queries, out_file
        )

    # reset max_tokens
    tabmem.config.max_tokens = prev_max_tokes

    # parse responses
    test_tokens = [x[:num_digits] for x in test_suffixes]
    response_tokens = [x[:num_digits] for x in responses]

    # count number of exact matches
    num_exact_matches = np.sum(np.array(test_tokens) == np.array(response_tokens))

    # count the number of exact matches using the most common first token
    num_exact_matches_most_common = np.sum(
        np.array(response_tokens) == most_common_first_token
    )

    # print result
    print(
        bcolors.BOLD
        + "First Token Test: "
        + bcolors.ENDC
        + bcolors.Black
        + f"{num_exact_matches}/{num_queries} exact matches.\n"
        + bcolors.ENDC
        + bcolors.BOLD
        + "First Token Test Baseline (Matches of most common first token): "
        + bcolors.ENDC
        + f"{num_exact_matches_most_common}/{num_queries}."
    )


####################################################################################
# Zero-Knowledge Sampling
####################################################################################


def build_sample_prompt(messages):
    prompt = ""
    for m in messages:
        if m["role"] == "user":
            prompt += m["content"] + "\n"
        elif m["role"] == "assistant":
            prompt += "Random Sample: " + m["content"] + "\n\n"
    prompt += "Random Sample: "
    return prompt


def sample(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    num_queries: int,
    few_shot_csv_files: list[str] = DEFAULT_FEW_SHOT_CSV_FILES,
    cond_feature_names: list[str] = [],
    drop_invalid_responses: bool = True,
    print_invalid_responses: bool = False,
    out_file=None,
    system_prompt: str = "default",
):
    """zero-shot sampling from the csv file, using few-shot examples from other csv files."""
    llm = __llm_setup(llm)
    few_shot_csv_files = __validate_few_shot_files(csv_file, few_shot_csv_files)

    if system_prompt == "default":  # default system prompt?
        system_prompt = tabmem.config.system_prompts["sample"]

    if not llm.chat_mode:  # wrap base model to take chat queries
        llm = ChatWrappedLLM(llm, build_sample_prompt, ends_with="\n\n")

    # run the test
    _, _, responses = feature_values_chat_completion(
        llm,
        csv_file,
        system_prompt,
        num_queries,
        few_shot_csv_files,
        cond_feature_names,
        add_description=True,
        out_file=None,
    )

    if len(cond_feature_names) > 0:
        pass
        # TODO handle the condtional case!

    # parse the model responses in a dataframe
    feature_names = utils.get_feature_names(csv_file)
    response_df = utils.parse_feature_stings(responses, feature_names)

    # get the indices of the rows with more than 50% NaN's
    nan_rows = response_df[
        response_df.isna().sum(axis=1) > 0.5 * len(feature_names)
    ].index.to_list()

    if print_invalid_responses:
        for idx in nan_rows:
            tabmem.llm.pretty_print_response(responses[idx])

    if drop_invalid_responses:
        response_df.drop(nan_rows, axis=0, inplace=True)

    # save the dataframe with the final samples
    if out_file is not None:
        print(out_file)
        response_df.to_csv(out_file, index=False)

    return response_df
