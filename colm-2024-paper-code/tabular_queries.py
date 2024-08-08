#
# Few-shot classification of tabular data with LLMs.
#
# The functions here take tabular datasets as numpy arrays and send
# messages to a tabmemcheck.LLM_Interface.
#

import numpy as np

import tabmemcheck
from tabmemcheck import LLM_Interface, send_chat_completion


####################################################################################
# Tabular chat completion fit-predict function
####################################################################################


def tabular_chat_fit_predict_fn_factory(
    llm, feature_names, target_name, system_prompt, **kwargs
):
    # the fit_predict function that we return to the user
    def fit_predict(X_train, y_train, testpoint):
        prediction = send_tabular_chat_completion(
            llm,
            X_train,
            y_train,
            testpoint,
            feature_names,
            target_name,
            system_prompt,
            **kwargs,
        )
        return 0  # TODO: return the prediction but in guarnateed format?

    return fit_predict


####################################################################################
# Send chat completion messages for tabular data
####################################################################################


def send_tabular_chat_completion(
    llm: LLM_Interface,
    X_train: np.array,
    y_train: np.array,
    testpoint: np.array,
    feature_names: list,
    target_name: str,
    system_prompt: str,
    target_name_location: str = "assistant",  # 'user' or 'assistant'
    optional_features=None,
    temperature=0.0,
    max_tokens=100,
):
    messages = [{"role": "system", "content": system_prompt}]
    for idx in range(X_train.shape[0]):
        messages.append(
            {
                "role": "user",
                "content": format_data_point(
                    X_train[idx, :],
                    feature_names,
                    optional_features=optional_features,
                    add_if_then=len(feature_names) != 0,
                ),
            }
        )
        # optionally, add the target name to the user message
        if target_name_location == "user" and len(target_name) > 0:
            messages[-1]["content"] = messages[-1]["content"] + f" {target_name} ="
        # the assistant reponse, optinally with target name
        if target_name_location == "assistant" and len(target_name) > 0:
            messages.append(
                {"role": "assistant", "content": f"{target_name} = {y_train[idx]}"}
            )
        else:
            messages.append({"role": "assistant", "content": f"{y_train[idx]}"})
    messages.append(
        {
            "role": "user",
            "content": format_data_point(
                testpoint,
                feature_names,
                optional_features=optional_features,
                add_if_then=len(feature_names) != 0,
            ),
        }
    )
    # optionally, add the target name to the user message
    if target_name_location == "user" and len(target_name) > 0:
        messages[-1]["content"] = messages[-1]["content"] + f" {target_name} ="

    tabmemcheck.temperature = (
        temperature  # TODO this really needs to be changed in the package
    )
    response = send_chat_completion(
        llm,
        messages,
        # temperature=temperature,
        max_tokens=max_tokens,
    )

    return response


#################################################################################################
# from numpy arrays to textual promts
#################################################################################################


def format_data_point(x, feature_names, optional_features=None, add_if_then=False):
    """X1 = 0.35, X2 = 0.82, X3 = 0.33, X4 = -1.30"""
    num_features = None
    if isinstance(
        x, np.ndarray
    ):  # support numpy arrays and lists (or does len work for the array, too?)
        x = np.atleast_1d(x.squeeze())
        num_features = x.shape[0]
    else:
        num_features = len(x)
    prompt = ""
    for idx in range(num_features):
        if len(feature_names) == 0:  # feature value
            prompt = prompt + f"{x[idx]}, "
        else:  # feature name = feature value, feature name = feature value, ...
            if (
                optional_features is not None
                and feature_names[idx] in optional_features
            ):  # optional features are only included if they are not zero
                if x[idx] == 0:
                    continue
            prompt = prompt + f"{feature_names[idx]} = {x[idx]}, "
    prompt = prompt[:-2]  # no comma at the end
    if add_if_then:
        prompt = "IF " + prompt + ", THEN"
    return prompt


#################################################################################################
# utility functions to parse the model output
#################################################################################################


def read_prefix_float(str, default=None):
    """Read a float from a maximum prefix of the string"""
    for i in reversed(range(len(str) + 1)):
        try:
            return float(str[:i].strip())
        except:
            pass
    if default is not None:
        print(f"String {str} does not have a prefix that is a float.")
        return default
    raise ValueError(f"String '{str}' does not have a prefix that is a float.")


def read_postfix_float(str, default=None):
    """Read a float from a maximum postifx of the string"""
    for i in range(len(str)):
        try:
            return float(str[i:].strip())
        except:
            pass
    if default is not None:
        print(f"String {str} does not have a postfix that is a float.")
        return default
    raise ValueError(f"String '{str}' does not have a postfix that is a float.")


def read_float(str, default=None):
    """read any float from the string. will work well if there is exactly a single float in the string."""
    for i in range(len(str)):
        for j in reversed(range(len(str) + 1)):
            try:
                return float(str[i:j].strip())
            except:
                pass
    if default is not None:
        print(f"String {str} does not contain a float.")
        return default
    raise ValueError(f"String '{str}' does not contain a float.")
