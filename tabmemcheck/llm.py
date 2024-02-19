#
# Functions to control the language model and send (chat) completions.
#
# Provides re-trying with exponential backoff and low-level logging of model responses.
#

from dataclasses import dataclass
from datetime import datetime

import os
import pickle
import time

import openai
from openai import OpenAI

import tiktoken

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

import tabmemcheck as tabmem

####################################################################################
# the interface to the language model, used by the test functions
####################################################################################


@dataclass
class LLM_Interface:
    """The interface to the language model."""

    # if true, the tests use the chat_completion function, otherwise the completion function
    chat_mode = False

    def completion(self, prompt, temperature, max_tokens):
        """Returns: The response (string)"""

    def chat_completion(self, messages, temperature, max_tokens):
        """Returns: The response (string)"""
        raise NotImplementedError


####################################################################################
# wrap a base language model to act as a chat completion model
####################################################################################


class ChatWrappedLLM(LLM_Interface):
    """Wrap a base language model (i.e. an LLM_Interface that only implements the completion method) to act as a chat completion model.

    The wrapped model take queries via the chat_completion interface. It transforms the messages list into a single textual prompt using the provided prompt_fn.
    """

    def __init__(self, llm, prompt_fn, ends_with: str = None):
        assert not llm.chat_mode, "The wrapped model must be a base model."
        self.llm = llm
        self.chat_mode = True
        self.wrapper_fn = prompt_fn
        self.ends_with = ends_with

    def chat_completion(self, messages, temperature, max_tokens):
        prompt = self.wrapper_fn(messages)
        # print(prompt)
        response = self.llm.completion(prompt, temperature, max_tokens)
        # print(response)
        if (
            self.ends_with is not None
        ):  # we frequently use '\n\n' as the end of the relevant part of the response
            if self.ends_with in response:
                response = response[: response.find(self.ends_with)]
        return response

    def __repr__(self) -> str:
        return self.llm.__repr__()


#################################################################################################
# openai
#################################################################################################


@dataclass
class OpenAILLM(LLM_Interface):
    client: OpenAI = None
    model: str = None

    def __init__(self, client, model=None):
        super().__init__()
        self.client = client
        self.model = model
        # auto-detect chat models
        if "gpt-3.5" in model or "gpt-4" in model:
            self.chat_mode = True

    @retry(
        retry=retry_if_not_exception_type(openai.BadRequestError),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(7),
        reraise=True,
    )
    def completion(self, prompt, temperature, max_tokens):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # we return the completion string or "" if there is an invalid response/query
        try:
            response = response.choices[0].text
        except:
            print(f"Invalid response {response}")
            response = ""
        return response

    # @retry(
    #    retry=retry_if_not_exception_type(openai.BadRequestError),
    #    wait=wait_random_exponential(min=1, max=60),
    #    stop=stop_after_attempt(7),
    #    reraise=True,
    # )
    def chat_completion(self, messages, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # we return the completion string or "" if there is an invalid response/query
        try:
            response = response.choices[0].message.content
        except:
            print(f"Invalid response {response}")
            response = ""
        return response

    def __repr__(self) -> str:
        return f"{self.model}"


def openai_setup(model=None):
    """Setup the openai api. Returns: LLM_Interface object."""
    client = OpenAI(
        api_key=(
            os.environ["OPENAI_API_KEY"] if "OPENAI_API_KEY" in os.environ else None
        ),
        organization=(
            os.environ["OPENAI_API_ORG"] if "OPENAI_API_ORG" in os.environ else None
        ),
        # timeout=20,
    )
    # the llm interface object
    return OpenAILLM(client, model)


#################################################################################################
# Gemini (requires pip install google-generativeai)
#################################################################################################


@dataclass
class GoogleGeminiLLM(LLM_Interface):
    def __init__(self, model: str):
        import google.generativeai as genai

        self.model = genai.GenerativeModel(model)
        self.chat_mode = True

    def chat_completion(self, messages, temperature, max_tokens):
        import google.generativeai as genai

        # convert messages from OpenAI format to Gemini format
        gemini_messages = []
        for message in messages:
            if message["role"] == "system":
                continue
            elif message["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [message["content"]]})
            elif message["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [message["content"]]})
            else:
                raise ValueError("Unknown message role: {}".format(message["role"]))

        # print(messages)
        # print(gemini_messages)
        # send messages to the model
        response = self.model.generate_content(
            gemini_messages,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1, max_output_tokens=max_tokens, temperature=temperature
            ),
        )

        # return response text
        # print(response.prompt_feedback)
        # print(response.parts)
        try:
            response = response.text
        except:
            print(f"Gemini: Invalid response with parts {response.parts}.")
            response = ""
        return response

    def __repr__(self) -> str:
        return f"{self.model}"


def gemini_setup(model: str = None, api_key: str = None):
    import google.generativeai as genai

    genai.configure(
        api_key=(
            os.environ["GEMINI_API_KEY"] if "GEMINI_API_KEY" in os.environ else api_key
        )
    )
    if model is not None:
        return GoogleGeminiLLM(model)
    return None


#################################################################################################
# huggingface transformers
#################################################################################################

"""
from transformers import AutoTokenizer, pipeline

HF_TOKENIZER = None
HF_PIPE = None


def hf_setup(model=None):
    global HF_ACCESS_TOKEN, HF_TOKENIZER, HF_PIPE
    os.environ["HF_ACCESS_TOKEN"] = "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"
    HF_TOKENIZER = AutoTokenizer.from_pretrained(model)
    HF_PIPE = pipeline(
        "text-generation", model=model, torch_dtype="auto", device_map="auto"
    )


def hf_completion(prompt, temperature, max_tokens):
    global HF_TOKENIZER, HF_PIPE


def hf_chat_completion(messages, temperature, max_tokens):
    global HF_TOKENIZER, HF_PIPE
    prompt = HF_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if temperature > 0.0:  # sampling
        outputs = HF_PIPE(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            eos_token_id=HF_TOKENIZER.eos_token_id,
        )
    else:  # no sampling
        outputs = HF_PIPE(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=HF_TOKENIZER.eos_token_id,
        )
    return outputs[0]["generated_text"][
        len(prompt) :
    ].strip()  # remove leading whitespaces
"""

#################################################################################################
# guidance
#################################################################################################


def guidance_llm_wrapper(llm):
    raise NotImplementedError


####################################################################################
# Send (chat) completion messages with retrying and logging.
#
# Logging works as follows. Either we specify a logfile, then we log to that file.
# Or we specify a logging task, then we log to taskname/taskname-{idx}.pkl
# where idx is the number of the query under that task.
# The taskname will typically be the name of the dataset.
####################################################################################


def set_logging_task(task):
    """Set the global task for logging."""
    config = tabmem.config
    config.current_logging_task = task
    config.current_logging_folder = f"chatlogs/{config.current_logging_task}"
    config.logging_task_index = 0
    # check if the folder exists, if not create it
    if not os.path.exists(config.current_logging_folder):
        os.makedirs(config.current_logging_folder)


def log(messages, response, logfile):
    """Log the messages and response."""
    config = tabmem.config
    # logging of the raw response object (e.g. the full openai response)
    if logfile is None:
        if config.current_logging_task is not None:
            logfile = f"{config.current_logging_task}/{config.current_logging_task}-{config.logging_task_index}.pkl"
            config.logging_task_index += 1
    if logfile is not None:
        with open(f"chatlogs/{logfile}", "wb+") as f:
            pickle.dump((messages, response), f)


def read_chatlog(taskname, root="chatlogs", min_files=-1):
    """A chaglog is a sequnces of files 'taskname-{idx}.pkl' that contain message-response pairs"""
    messages = []
    responses = []
    task_dir = os.path.join(root, taskname)
    # list all the files in task_dir
    task_files = os.listdir(task_dir)
    for idx in range(10000):
        # fname = f"{taskname}-{idx}.pkl"
        fsuffix = f"-{idx}.pkl"
        fname = [f for f in task_files if f.endswith(fsuffix)]
        if len(fname) > 1:
            print(f"Warning: found more than one file with suffix {fsuffix}")
        if len(fname) > 0:
            fname = fname[0]
            try:
                with open(os.path.join(task_dir, fname), "rb") as f:
                    message, response = pickle.load(f)
                    messages.append(message)
                    responses.append(response)
            except:
                print(f"Failed to read {fname}")
                messages.append(None)
                responses.append(None)
        elif (
            len(messages) > min_files
        ):  # if we already have enough results, then this is the end
            break
        else:
            print(f"File {fname} not found.")
            messages.append(None)
            responses.append(None)
    return messages, responses


def send_completion(llm: LLM_Interface, prompt, max_tokens=None, logfile=None):
    config = tabmem.config
    if max_tokens is None:
        max_tokens = config.max_tokens
    response = llm.completion(prompt, config.temperature, max_tokens)
    # logging
    log(prompt, response, logfile)
    # printing
    if config.print_prompts or config.print_next_prompt:
        pretty_print_completion(prompt, response)
    elif config.print_responses:
        pretty_print_response(response)
    # reset print_next_prompt
    config.print_next_prompt = False
    # return string response
    return response


def send_chat_completion(llm: LLM_Interface, messages, max_tokens=None, logfile=None):
    """Send chat completion with retrying and logging.

    Returns: The response (string))"""
    config = tabmem.config
    if max_tokens is None:
        max_tokens = config.max_tokens
    response = llm.chat_completion(messages, config.temperature, max_tokens)
    if config.sleep > 0.0:
        time.sleep(config.sleep)
    # logging
    log(messages, response, logfile)
    # printing
    if config.print_prompts or config.print_next_prompt:
        pretty_print_messages(messages)
    if config.print_prompts or config.print_responses or config.print_next_prompt:
        pretty_print_response(response)
    # reset print_next_prompt
    config.print_next_prompt = False
    # return string response
    return response


#################################################################################################
# misc
#################################################################################################


def num_tokens_from_string(string: str, model_name: str = None) -> int:
    """Returns the number of tokens in a text string."""
    # if the user did not specify the encoding, take the maximum over gpt 3.5 and gpt 4
    # TODO in the future do this for the specified llm, once we have it in the codebase
    if model_name is None:
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens_gpt_4 = len(encoding.encode(string))
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens_gpt_3_5 = len(encoding.encode(string))
        return max(num_tokens_gpt_4, num_tokens_gpt_3_5)
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


#################################################################################################
# pretty printing
#################################################################################################


def pretty_print_completion(prompt, response):
    """Print a plain language model completion in a nice format"""
    print(
        bcolors.Green + prompt + bcolors.ENDC + bcolors.Purple + response + bcolors.ENDC
    )


def pretty_print_messages(messages):
    """Prints openai chat messages in a nice format"""
    for message in messages:
        print(
            bcolors.BOLD
            + message["role"].capitalize()
            + ": "
            + bcolors.ENDC
            + bcolors.Green
            + message["content"].strip()
            + bcolors.ENDC
        )


def pretty_print_response(response):
    """Prints openai chat response in a nice format"""
    print(
        bcolors.BOLD
        + "Response: "
        + bcolors.ENDC
        + bcolors.Purple
        + response
        + bcolors.ENDC,
    )


#################################################################################################
# color codes to print with color in the console (from https://gist.github.com/vratiu/9780109)
#################################################################################################


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Regular Colors
    Black = "\033[0;30m"  # Black
    Red = "\033[0;31m"  # Red
    Green = "\033[0;32m"  # Green
    Yellow = "\033[0;33m"  # Yellow
    Blue = "\033[0;34m"  # Blue
    Purple = "\033[0;35m"  # Purple
    Cyan = "\033[0;36m"  # Cyan
    White = "\033[0;37m"  # White

    # Background
    On_Black = "\033[40m"  # Black
    On_Red = "\033[41m"  # Red
    On_Green = "\033[42m"  # Green
    On_Yellow = "\033[43m"  # Yellow
    On_Blue = "\033[44m"  # Blue
    On_Purple = "\033[45m"  # Purple
    On_Cyan = "\033[46m"  # Cyan
    On_White = "\033[47m"  # White
