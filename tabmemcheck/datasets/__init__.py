from . import transform as transform
from . import load as load

# utility functions to load common datasets with default transformations specified in the package
from .load import (
    load_dataset,
    load_iris,
    # load_titanic,
    load_openml_diabetes,
    load_adult,
    load_housing,
)
