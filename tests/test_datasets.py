import tabmemcheck

from tabmemcheck import datasets

from testutils import TestLLM


def test_transform():
    csv_file = "adult-train.csv"

    datasets.load_dataset(csv_file, "adult.yaml", transform="original")
    datasets.load_dataset(csv_file, "adult.yaml", transform="perturbed")
    datasets.load_dataset(csv_file, "adult.yaml", transform="task")
    datasets.load_dataset(csv_file, "adult.yaml", transform="statistical")
