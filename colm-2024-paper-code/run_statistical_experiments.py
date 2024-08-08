#
# Run as 'python run_statistical_experiments.py'
#

import yaml

import tabmemcheck

import tabular_queries
import statutils


# main entry point
if __name__ == "__main__":
    print("Starting experiments.")

    ####################################################################################################
    # experiments with synthetic statistical data
    ####################################################################################################

    llm = tabmemcheck.openai_setup("gpt40125", azure=True)

    system_prompts = {}
    with open("config/tabular-system-prompts.yaml", "r") as f:
        system_prompts = yaml.load(f, Loader=yaml.FullLoader)

    # the experiments
    experiments = {}
    for dim in [8]:
        for idx in range(1000):
            experiments[f"linear-classification-d={dim}-replication={idx}-seed-0"] = {
                "csv-file": f"datasets/synthetic/linear-classification-d={dim}-replication={idx}.csv",
                "seed": 0,
            }

    # run the experiments
    for exp_name, experiment in experiments.items():
        # the random seed
        # controls the few-shot samples
        seed = experiment["seed"]

        # read parameters from experiment
        print("Running experiment: ", exp_name)
        print("Experiment parameters: ", experiment)
        print("Seed: ", seed)

        # load the dataset in the desired version
        df = tabmemcheck.datasets.load_dataset(
            experiment["csv-file"],
            "plain",
        )
        print(df.head())

        # the names of the features and the target
        feature_names, target_name = df.columns.tolist()[:-1], df.columns.tolist()[-1]

        # the data - conversion to string preserves the formatting
        X_data, y_data = (
            df[feature_names].astype("str").values,
            df[target_name].astype("str").values,
        )

        # the system prompt is always binary classification
        system_prompt = system_prompts["general"]["binary_classification"]

        # logging of prompts and responses
        tabmemcheck.set_logging_task(exp_name)
        tabmemcheck.config["temperature"] = 0.0
        tabmemcheck.config["print_prompts"] = True
        tabmemcheck.config["print_responses"] = True

        # send queries to the LLM
        fit_predict_fn = tabular_queries.tabular_chat_fit_predict_fn_factory(
            llm,
            feature_names,
            target_name,
            system_prompt,
            target_name_location="user",
            max_tokens=1,
        )

        statutils.loo_eval(
            X_data,
            y_data,
            fit_predict_fn,
            few_shot=10,
            max_points=1,
            stratified=True,  # set false for regression
            random_state=seed,
        )

    print("All experiments done.")
