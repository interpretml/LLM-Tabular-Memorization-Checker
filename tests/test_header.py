"""
if __name__ == "__main__":
    csv_file = CSVFile("csv/california-housing.csv")
    print(csv_file.get_dataset_name())
    print(csv_file.get_feature_names())
    print(csv_file.load_rows()[1])

    df = csv_file.load_df()
    # randomly permute the columns
    df = df.sample(frac=1, axis=1)
    csv_file = CSVFile.from_df(df, "California Housing")
    print(csv_file.get_dataset_name())
    print(csv_file.get_feature_names())
    print(csv_file.load_rows()[1])
"""

import tabmemcheck as tabmem

from tutils import TestLLM


def test_header():
    csv_file = "csv/openml-diabetes.csv"
    llm = TestLLM()
    chat_llm = TestLLM()
    chat_llm.chat_mode = True
    tabmem.config.print_prompts = True
    tabmem.header_test(csv_file, llm)
    tabmem.header_test(csv_file, chat_llm)
