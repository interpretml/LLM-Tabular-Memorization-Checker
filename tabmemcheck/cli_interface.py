import os

import yaml
import copy

import importlib.resources as resources

import tabmemcheck as tabmem
import tabmemcheck.llm as llm_utils
import tabmemcheck.utils as utils


from tabmemcheck.row_independence import row_independence_test


def main():
    # parse args
    import argparse

    parser = argparse.ArgumentParser(
        prog="tabmemcheck",
        description=f"Testing Language Models for Memorization of Tabular Data.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"Program Version {tabmem.__version__}",
    )

    # required args
    tasks = {
        "all": "Automatically perform a series of tests and print a small report.",
        # "predict": "Few-shot prediction of a feature in the csv file.",
        "feature-names": "Does the LLM know the feature names from the top row of the csv file?",
        # "feature-values": "Does the LLM produce feature values of the same format as in the csv file?",
        # "mode": "Mode test for unconditional distribution modelling.",
        # "ordered-completion": "Feature completion, respecting the order of the features in the csv file.",
        "header": "Header test for memorization.",
        "row-completion": "Row completion test for memorization.",
        "feature-completion": "Feature completion test for memorization.",
        "first-token": "First token test for memorization.",
        "row-independence": "Row independence test, using a gradient boosted tree and logistic regression.",
        "sample": "Zero-knowledge samples from the parametric knowledge of the LLM.",  # For conditional sampling, use the parameter --cond.
    }

    parser.add_argument("csv", type=str, help="A csv file.")
    parser.add_argument(
        "task",
        type=str,
        choices=list(tasks.keys()),
        help="The task that should be performed with the csv file.\n  - "
        + "\n  - ".join([k + ": " + v for k, v in tasks.items()]),
        metavar="task",
    )

    # test parameters
    parser.add_argument("--header-length", type=int, default=500)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-prefix-rows", type=int, default=None)
    parser.add_argument("--num-prefix-features", type=int, default=None)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument(
        "--few-shot",
        "--names-list",
        nargs="*",
        default=tabmem.functions.DEFAULT_FEW_SHOT_CSV_FILES,
    )
    # the --cond parameter takes a list of strings
    parser.add_argument("--cond", nargs="*", default=[])

    # openai api args
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-1106",
        help="Default: gpt-3.5-turbo-1106",
    )
    # parser.add_argument(
    #    "--api", type=str, choices=["openai", "hf", "guidance"], default="openai"
    # )

    # LLM
    parser.add_argument("--temperature", type=float, default=0.0, help="Default: 0.0")

    # misc args
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to save the results (a filename).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Number of seconds to sleep after each query to the LLM.",
    )
    # print prompts & responses
    # parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--pp", default=False, action="store_true", help="Print prompts and responses."
    )
    parser.add_argument(
        "--pr", default=False, action="store_true", help="Print responses only."
    )

    args = parser.parse_args()

    # populate the tabmem config with values specified in the args
    tabmem.config.print_responses = args.pr
    if args.pp:
        tabmem.config.print_prompts = True
        tabmem.config.print_responses = True
    tabmem.config.temperature = args.temperature
    tabmem.config.sleep = args.sleep

    # initialize the llm from the command line args
    # if args.api == "openai":
    llm = llm_utils.openai_setup(model=args.model)
    # else:
    #    raise NotImplementedError

    # args.csv can be a single csv file or a directory
    csv_files = [args.csv]
    if os.path.isdir(args.csv):
        csv_files = [
            os.path.join(args.csv, file)
            for file in os.listdir(args.csv)
            if file.endswith(".csv")
        ]

    # --few-shot can be an integer or a list of csv files
    few_shot = args.few_shot
    if few_shot is not None:
        if len(few_shot) == 1:
            try:  # attempt to convert to int. if it fails, assume it is a csv file
                few_shot = int(few_shot[0])
            except ValueError:
                pass

    # run the specified task for all the csv files
    for csv_file in csv_files:
        print(
            llm_utils.bcolors.BOLD + f"File: " + llm_utils.bcolors.ENDC + f"{csv_file}"
        )
        # make sure that the current csv file is not contained in the few-shot list
        few_shot_csv_files = copy.deepcopy(few_shot)
        if isinstance(few_shot, list):
            few_shot_csv_files = [
                file
                for file in few_shot
                if not utils.get_dataset_name(csv_file) in file
            ]
            if len(few_shot_csv_files) < len(few_shot):
                middle = len(few_shot_csv_files) // 2
                few_shot_csv_files = (
                    few_shot_csv_files[:middle]
                    + ["openml-diabetes.csv"]
                    + few_shot_csv_files[middle:]
                )
                print(
                    "INFO: The csv file was contained in the few-shot list, replaced it with openml-diabetes.csv"
                )

        ##############################
        # All the tests
        ##############################
        if args.task == "all":
            tabmem.run_all_tests(
                csv_file,
                llm,
                few_shot_csv_files=few_shot_csv_files,
            )

        ##############################
        # Feature Names
        ##############################
        if args.task == "feature-names":
            tabmem.feature_names_test(
                csv_file,
                llm,
                num_prefix_features=args.num_prefix_features,
                few_shot_csv_files=few_shot_csv_files,
            )

        ##############################
        # Zero-Knowledge Sampling
        ##############################
        elif args.task == "sample":
            tabmem.sample(
                csv_file,
                llm,
                num_queries=args.num_queries,
                few_shot_csv_files=few_shot_csv_files,
                cond_feature_names=args.cond,
                out_file=args.out,
            )

        ##############################
        # Header Test
        ##############################
        elif args.task == "header":
            tabmem.header_test(csv_file, llm, few_shot_csv_files=few_shot_csv_files)

        ##############################
        # Row Completion
        ##############################
        elif args.task == "row-completion":
            if args.num_prefix_rows is None:
                args.num_prefix_rows = 15
            if args.few_shot is None or isinstance(args.few_shot, list):
                args.few_shot = 7
            tabmem.row_completion_test(
                csv_file,
                llm,
                num_queries=args.num_queries,
                num_prefix_rows=args.num_prefix_rows,
                few_shot=args.few_shot,
                out_file=args.out,
            )

        ##############################
        # Feature Completion
        ##############################
        elif args.task == "feature-completion":
            tabmem.feature_completion_test(
                csv_file,
                llm,
                feature_name=args.target,
                num_queries=args.num_queries,
                out_file=args.out,
            )

        ##############################
        # First Token
        ##############################
        elif args.task == "first-token":
            if args.num_prefix_rows is None:
                args.num_prefix_rows = 15
            if args.few_shot is None or isinstance(args.few_shot, list):
                args.few_shot = 7
            tabmem.first_token_test(
                csv_file,
                llm,
                num_queries=args.num_queries,
                num_prefix_rows=args.num_prefix_rows,
                few_shot=args.few_shot,
                out_file=args.out,
            )

        ##############################
        # Row Independence
        ##############################
        elif args.task == "row-independence":
            row_independence_test(csv_file)

    #        elif args.task == "completion":
    #            tests.conditional_completion_test(
    #                csv_file,
    #                config["sample"],
    #                feature_name=args.target,
    #                num_queries=args.num_queries,
    #                few_shot=few_shot_csv_files,
    #                out_file=args.out,
    #            )
    #        elif args.task == "predict":
    # set the callback
    # experiment_utils.on_response_callback_fn = analysis.csv_match_callback(
    #    csv_file
    # )
    #            tests.predict(
    #                csv_file,
    #                config["predict"],
    #                args.target,
    #                num_queries=args.num_queries,
    #                # TODO few_shot=csv_file_few_shot,
    #            )
    #        elif args.task == "format":
    #            tests.csv_format_test(csv_file, system_prompt=config["generic-csv-format"])

    #        elif args.task == "ordered-completion":
    #            tests.ordered_completion(
    #                csv_file,
    #                config["feature-completion"],
    #                feature_name=args.target,
    #                num_queries=args.num_queries,
    #                few_shot=few_shot_csv_files,
    #                out_file=args.out,
    #            )

    exit(0)


if __name__ == "__main__":
    main()
