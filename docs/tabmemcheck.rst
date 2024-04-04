Documentation
=============

This is the documentation for the tabmemcheck package.

Tests for tabular datasets (based on csv files)
-----------------------------------------------

.. automodule:: tabmemcheck
   :members: run_all_tests, header_test, feature_names_test, row_completion_test, feature_completion_test, first_token_test, sample
   :show-inheritance:

Dataset loading (original, perturbed, task, statistical)
--------------------------------------------------------

.. automodule:: tabmemcheck.datasets
   :members: load_dataset, load_iris, load_wine, load_adult, load_housing, load_openml_diabetes
   :show-inheritance:


LLM Interface
----------------------

.. automodule:: tabmemcheck
   :members: LLM_Interface, openai_setup, send_chat_completion, send_completion, set_logging_task, read_chatlog
   :show-inheritance:

Analysis
------------------------

.. automodule:: tabmemcheck.analysis
   :members:
   :show-inheritance:


Utilities
------------------------

.. autoclass:: tabmemcheck.utils
   :members:
   :show-inheritance:


