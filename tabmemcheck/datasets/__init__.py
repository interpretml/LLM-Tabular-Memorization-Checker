from . import transform as transform
from . import load as load

# utility functions to load common datasets with default transformations specified in the package
from .load import (
    load_dataset,
    load_iris,
    load_wine,
    load_adult,
    load_housing,
    load_openml_diabetes,
)
