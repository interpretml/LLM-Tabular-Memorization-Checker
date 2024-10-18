# llm interface
from .llm import (
    LLM_Interface,
    openai_setup,
    gemini_setup,
    send_chat_completion,
    send_completion,
)

# logging
from .llm import set_logging_task, read_chatlog

# the different tests
from .functions import (
    run_all_tests,
    header_test,
    feature_names_test,
    feature_values_test,
    dataset_name_test,
    row_completion_test,
    feature_completion_test,
    first_token_test,
    sample,
)

# data loading and transform
import tabmemcheck.datasets as datasets

from .version import __version__


def __load_default_system_prompts():
    """Load the default system prompts from the resources folder."""
    import importlib.resources as resources
    import yaml

    with resources.open_text(
        "tabmemcheck.resources.config", "system-prompts.yaml"
    ) as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)
    return prompts


# global config object for the module
class __DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]


config = __DotDict({})

# default system prompts from yaml file
config.system_prompts = __load_default_system_prompts()

# default llm options
config.temperature = 0
config.max_tokens = 1000
config.sleep = 0.0  # amount of time to sleep after each query to the llm

# csv file loading options
config.csv_max_rows = 100000        # maximum number of rows to load from a csv file

# how to display test output. "cmd" or "html" for jupyter notebook html display
config.display = "cmd"

# default: no prompt/response logging
config.current_logging_task = None
config.current_logging_folder = None
config.logging_task_index = 0

# default: no prompt printing
config.print_prompts = False
config.print_responses = False
config.print_next_prompt = False
