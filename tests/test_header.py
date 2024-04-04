import tabmemcheck
from testutils import TestLLM



def test_header():
    csv_file = "adult-train.csv"
    llm = TestLLM()
    chat_llm = TestLLM()
    chat_llm.chat_mode = True
    tabmemcheck.config.print_prompts = True
    tabmemcheck.header_test(csv_file, llm)
    tabmemcheck.header_test(csv_file, chat_llm)
