# Code for the paper "Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models"

Here we provide the code to replicate the COLM'24 [paper](https://arxiv.org/abs/2404.06209) "Elephants Never Forget: Memorization and Learning of Tabular Data in Large Language Models".

The code is organized as follows.

- The files ```run_tabular_experiments.py``` ```run_time_series_experiments.py``` and ```run_statistical_experiments.py``` run the different experiments, that is sending queries to the LLM.
- The LLM queries are saved to disk and analyzed in Jupyter Notebooks. These are contained in the ```notebooks``` folder. The notebooks generate the figures and tables in the paper.
- The memorization tests can be directly performed with the ```tabmemcheck``` package, see the notebook ```memorization-tests.ipynb```
- The ```datasets``` folder contains CSV files.
- The ```preprocessing``` folder contains notebooks that create the ACS Income, ACS Travel and ICU datasets.
- The ```config``` folder contains prompt configurations and YAML files that specify the different dataset transforms.
- The environment used to run the experiments is given in ``'environment.yml```

# Citation

```
@article{bordt2024colm,
  title={Elephants Never Forget: Memorization and Learning of Tabular Data in
  Large Language Models},
  author={Bordt, Sebastian and Nori, Harsha and Rodrigues, Vanessa and Nushi, Besmira and Caruana, Rich},
  journal={Conference on Language Modeling (COLM)},
  year={2024}
}
```
