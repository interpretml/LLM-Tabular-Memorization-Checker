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
        # replace the dataset with iris or adult 
        few_shot_csv_files = [
            x for x in few_shot_csv_files if utils.get_dataset_name(x) != dataset_name
        ]
        if 'iris' in dataset_name:
            few_shot_csv_files.append("adult-train.csv")
        else:
            few_shot_csv_files.append("iris.csv")
        print(
                bcolors.BOLD
                + "Info: "
                + bcolors.ENDC
                + f"Exchanged a few-shot datasets because its name is similar to the dataset being tested."
            )
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
    unique_feature: str = None,
):
    """Run different tests for memorization and prior experience with the content of the csv file.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param few_shot_csv_files: A list of other csv files to be used as few-shot examples.
    :param unique_feature: The name of the feature to be used for the feature completion test.
    """
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
    feature_completion_test(csv_file, llm, num_queries=25, feature_name=unique_feature)
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
    """Test if the model knows the names of the features in a csv file.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param num_prefix_features: The number of features given to the model as part of the prompt (defaults to 1/4 of the features).
    :param few_shot_csv_files: A list of other csv files to be used as few-shot examples.
    :param system_prompt: The system prompt to be used.
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
    verbose: bool = True,
):
    """Header test for memorization.

    We split the csv file at random positions in rows split_rows and performs 1 query for each split. Then we compare the best completion with the actual header.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param split_rows: The rows at which the csv file is split for the test.
    :param completion_length: The length of the completions in the few-shot examples (reduce for LLMs with small context windows).
    :param few_shot_csv_files: A list of other csv files to be used as few-shot examples.
    :param system_prompt: The system prompt to be used.

    :return: The header prompt, the actual header completion, and the model response.
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
    header_prompt, llm_completion = None, None
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
            header_prompt = prefixes[0]
            llm_completion = response
            header_completion = data[offset : offset + len(llm_completion)]

    if verbose:  # print test result to console
        print(
            bcolors.BOLD
            + "Header Test: "
            + bcolors.ENDC
            + bcolors.Black
            + header_prompt
            + utils.levenshtein_cmd(header_completion, llm_completion)
            + bcolors.ENDC
            + bcolors.BOLD
            + "\nHeader Test Legend:  "
            + bcolors.ENDC
            + "Prompt "
            + bcolors.Green
            + "Correct "
            + bcolors.Red
            + "Incorrect "
            + bcolors.ENDC
            + bcolors.Purple
            + "Missing"
            + bcolors.ENDC
        )

    return header_prompt, header_completion, llm_completion


####################################################################################
# Row Completion
####################################################################################


def row_completion_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    num_prefix_rows=10,
    num_queries=25,
    few_shot=7,
    out_file=None,
    system_prompt: str = "default",
    print_levenshtein: bool = True,
):
    """Row completion test for memorization. The test resports the number of correctly completed rows.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param num_prefix_rows: The number of rows given to the model as part of the prompt.
    :param num_queries: The number of rows that we test the model on.
    :param few_shot: The number of few-shot examples to be used.
    :param out_file: Optionally save all queries and responses to a csv file.
    :param system_prompt: The system prompt to be used.
    :param print_levenshtein: Print a visulization of the levenshtein distance between the model responses and the actual rows.

    :return: the rows, the model responses.
    """
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
        _, test_suffixes, responses = row_chat_completion(
            llm,
            csv_file,
            system_prompt,
            num_prefix_rows,
            num_queries,
            few_shot,
            out_file,
            print_levenshtein,
        )
    else:
        _, test_suffixes, responses = row_completion(
            llm, csv_file, num_prefix_rows, num_queries, out_file, print_levenshtein=print_levenshtein
        )

    # count the number of verbatim completed rows
    num_exact_matches = 0
    for test_suffix, response in zip(test_suffixes, responses):
        if test_suffix.strip() in response.strip():
            num_exact_matches += 1

    # the statistical test using the levenshtein distance. taken out of current version although it seems to work in practice.
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

    return test_suffixes, responses


####################################################################################
# Feature Completion
####################################################################################


def feature_completion_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    feature_name: str = None,
    num_queries=25,
    few_shot=5,
    out_file=None,
    system_prompt: str = "default",
):
    """Feature completion test for memorization. The test resports the number of correctly completed features.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param feature_name: The name of the feature to be used for the test.
    :param num_queries: The number of feature values that we test the model on.
    :param few_shot: The number of few-shot examples to be used.
    :param out_file: Optionally save all queries and responses to a csv file.
    :param system_prompt: The system prompt to be used.

    :return: the feature values, the model responses.
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
    response_df = utils.parse_feature_stings(
        responses, [feature_name], final_delimiter="\n"
    )
    test_suffix_df = utils.parse_feature_stings(
        test_suffixes, [feature_name], final_delimiter="\n"
    )

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

    return test_suffix_df[feature_name].to_list(), response_df[feature_name].to_list()


####################################################################################
# First Token Test
####################################################################################


def first_token_test(
    csv_file: str,
    llm: Union[LLM_Interface, str],
    num_prefix_rows=10,
    num_queries=25,
    few_shot=7,
    out_file=None,
    system_prompt: str = "default",
):
    """First token test for memorization. We ask the model to complete the first token of the next row of the csv file, given the previous rows. The test resports the number of correctly completed tokens.

    Note that the ''first token'' is not actually the first token produced by the llm, but consists of the first n digits of the row. The number of digits is determined by the function build_first_token.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param num_prefix_rows: The number of rows given to the model as part of the prompt.
    :param num_queries: The number of rows that we test the model on.
    :param few_shot: The number of few-shot examples to be used.
    :param out_file: Optionally save all queries and responses to a csv file.
    :param system_prompt: The system prompt to be used.
    """
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
# Sampling
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
    """Ask the model to provide random samples from the csv file.

    :param csv_file: The path to the csv file.
    :param llm: The language model to be tested.
    :param num_queries: The desired number of samples.
    :param few_shot_csv_files: A list of other csv files to be used as few-shot examples.
    :param out_file: Optionally save all queries and responses to a csv file.
    :param system_prompt: The system prompt to be used.
    """
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
        raise NotImplementedError("Conditional sampling not yet supported.")
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
            tabmem.llm.print_response(responses[idx])

    if drop_invalid_responses:
        response_df.drop(nan_rows, axis=0, inplace=True)

    # save the dataframe with the final samples
    if out_file is not None:
        print(out_file)
        response_df.to_csv(out_file, index=False)

    return response_df
