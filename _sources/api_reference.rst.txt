API Reference
=============

Tests for tabular datasets (based on csv files)
-----------------------------------------------

.. automodule:: tabmemcheck
   :members: run_all_tests, header_test, feature_names_test, row_completion_test, feature_completion_test, first_token_test, sample
   :show-inheritance:

Tabular dataset loading (original, perturbed, task, statistical)
----------------------------------------------------------------

.. automodule:: tabmemcheck.datasets
   :members: load_dataset, load_iris, load_wine, load_adult, load_housing, load_openml_diabetes
   :show-inheritance:


LLM
---

.. automodule:: tabmemcheck
   :members: LLM_Interface, openai_setup, send_chat_completion, send_completion
   :show-inheritance:

Analysis
------------------------

.. automodule:: tabmemcheck.analysis
   :members: find_matches, is_in_df, build_first_token, find_most_unique_feature
   :show-inheritance:


Utilities
------------------------

.. automodule:: tabmemcheck.utils
   :members: get_dataset_name, get_delimiter, get_feature_names, load_csv_df, load_csv_rows, load_csv_string, load_csv_array, load_samples, parse_feature_string, parse_feature_stings, levenshtein_cmd, levenshtein_html
   :show-inheritance:


