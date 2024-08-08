#
# Run as 'python run_tabular_experiments.py'
#

import yaml
from pathlib import Path

from sklearn.model_selection import train_test_split

import tabmemcheck

import tabular_queries
import statutils

import os
from together import Together


# main entry point
if __name__ == "__main__":
    print("Starting experiments.")

    ####################################################################################################
    # chat model leave-one-out evaluation with tabular data
    ####################################################################################################

    #llm = tabmemcheck.openai_setup(
    #    "gpt-3.5-turbo-0125"
    #)  # gpt-3.5-turbo-0125, gpt-3.5-turbo-16k-0613 gpt-4-0613 gpt-4-0125-preview


    client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
    llm = tabmemcheck.llm.OpenAILLM(client=client, model="google/gemma-2-27b-it", chat_mode=True) # Meta-Llama-3.1-70B-Instruct-Turbo google/gemma-2-27b-it

    system_prompts = {}
    with open("config/tabular-system-prompts.yaml", "r") as f:
        system_prompts = yaml.load(f, Loader=yaml.FullLoader)

    # datasets
    with open("datasets.yaml") as file:
        datasets = yaml.load(file, Loader=yaml.FullLoader)["datasets"]

    versions = ["original", "perturbed", "task", "statistical"]

    # construct the experiments
    experiments = {}
    for csv_file, yaml_file in datasets:
        for version in versions:
            exp_basename = Path(yaml_file).stem
            n_samples = tabmemcheck.datasets.load_dataset(
                csv_file, yaml_file, "original"
            ).shape[
                0
            ]  # number of samples in the dataset
            if n_samples > 100000000:
                experiments[f"{exp_basename}-{version}-0"] = {
                    "csv-file": csv_file,
                    "yaml-file": yaml_file,
                    "transform": version,
                    "system-prompt": exp_basename,
                    "seed": 0,
                }
            else:
                for seed in range(10):
                    experiments[f"{exp_basename}-{version}-{seed}"] = {
                        "csv-file": csv_file,
                        "yaml-file": yaml_file,
                        "transform": version,
                        "system-prompt": exp_basename,
                        "seed": seed,
                    }

    # run the experiments
    for exp_name, experiment in experiments.items():
        if (
            "iris" in exp_name or "icu" in exp_name 
        ):
            continue
        #if "adult-original" in exp_name or "adult-perturbed" in exp_name:
        #    continue
        if not "0" in exp_name:
            continue
        #if not ("0" in exp_name or "1" in exp_name or "2" in exp_name):
        #    continue

        # the random seed
        # controls dataset perturbations and the ordering of the few-shot samples
        seed = experiment["seed"]

        # read parameters from experiment
        print("Running experiment: ", exp_name)
        print("Experiment parameters: ", experiment)
        print("Seed: ", seed)

        # load the dataset in the desired version
        df = tabmemcheck.datasets.load_dataset(
            experiment["csv-file"],
            experiment["yaml-file"],
            experiment["transform"],
            seed=seed,
        )
        print(df.head())

        # the names of the features and the target
        feature_names, target_name = df.columns.tolist()[:-1], df.columns.tolist()[-1]

        # the data - conversion to string preserves the formatting
        X_data, y_data = (
            df[feature_names].astype("str").values,
            df[target_name].astype("str").values,
        )

        # for large datasets, perform a train-test split
        # this also shuffles the data which randomizes the test points
        if X_data.shape[0] > 1300:
            X_train, _, y_train, _ = train_test_split(
                X_data, y_data, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X_data, y_data

        # the system prompt
        prompt_version = "original"
        if experiment["transform"] in ["task", "statistical"]:
            prompt_version = experiment["transform"]
        system_prompt = system_prompts[experiment["system-prompt"]][prompt_version]
        if system_prompt in system_prompts["general"]:
            system_prompt = system_prompts["general"][system_prompt]

        # logging of prompts and responses
        tabmemcheck.set_logging_task(exp_name)
        tabmemcheck.config["temperature"] = 0.0
        # tabmemcheck.config["print_prompts"] = True
        tabmemcheck.config["print_responses"] = True

        # send queries to the LLM
        fit_predict_fn = tabular_queries.tabular_chat_fit_predict_fn_factory(
            llm,
            feature_names,
            target_name,
            system_prompt,
            target_name_location="user",
            max_tokens=10,
        )

        statutils.loo_eval(
            X_train,
            y_train,
            fit_predict_fn,
            few_shot=20,
            max_points=1000,
            stratified=True,  # set false for regression
            random_state=seed,
        )

    print("All experiments done.")
