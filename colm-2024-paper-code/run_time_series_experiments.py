#
# Run as 'python run_time_series_experiments.py'
#

import pandas as pd

import tabular_queries

import yaml


if __name__ == "__main__":
    # the experiments
    experiments = {
        "netflix": [2020, 2022],
        "nasdaqcomp": [2020, 2022],
        "msci-world": [2020, 2022],
        "usd-eur": [2020, 2022],
        "usd-yuan": [2020, 2022],
    }

    for time_series_name, years in experiments.items():
        for year in years:
            # the experiment
            exp_name = time_series_name + "-" + str(year)
            print("Running experiment: ", exp_name)

            # load the csv file
            csv_file = f"datasets/csv/time-series/{exp_name}.csv"
            df = pd.read_csv(csv_file)

            # load the config file
            yaml_file = f"config/time-series/{time_series_name}.yaml"
            try:
                with open(yaml_file, "r") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
            except FileNotFoundError:  # temperature has shared config file
                if "temperature" in exp_name:
                    yaml_file = f"config/time-series/temperature.yaml"
                    with open(yaml_file, "r") as f:
                        config = yaml.load(f, Loader=yaml.FullLoader)

            # the names of the features and the target
            feature_names, target_name = (
                df.columns.tolist()[:-1],
                df.columns.tolist()[-1],
            )

            # the data
            X_data, y_data = df[feature_names].values, df[target_name].values

            # length of the time series
            T_max = df.shape[0]

            # the system prompt
            system_prompt = config["system_prompt"]

            # replace {{year}} in the system prompt with the experiment year
            system_prompt = system_prompt.replace("{{year}}", str(year))

            # temperature experiments: {{location}} is the the city name
            if "{{location}}" in system_prompt:
                city_name = (
                    time_series_name[0].upper()
                    + time_series_name[1 : time_series_name.find("-")]
                )
                system_prompt = system_prompt.replace("{{location}}", city_name)

            # logging of messages and responses
            # the messages will be logged with {idx-20}
            tabular_queries.set_logging_task(exp_name)

            for idx in range(20, T_max):
                # we go sequentially trough the time series and use the previous 20 days as input to predict the current day
                X_train, y_train = X_data[idx - 20 : idx, :], y_data[idx - 20 : idx]
                testpoint, testlabel = X_data[idx, :], y_data[idx]

                tabular_queries.send_tabular_chat_completion(
                    X_train,
                    y_train,
                    testpoint,
                    feature_names,
                    target_name,
                    messages=[{"role": "system", "content": system_prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
