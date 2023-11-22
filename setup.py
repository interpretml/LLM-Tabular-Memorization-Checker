import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tabmemcheck",
    version="0.1.0",
    author="Sebastian Bordt, Harsha Nori, Rich Caruana",
    author_email="sbordt@posteo.de",
    description="Testing Language Models for Memorization of Tabular Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/LLM-Tabular-Memorization-Checker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=["tabmemcheck"],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": ["tabmemcheck=tabmemcheck.cli_interface:main"],
    },
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "xgboost",
        "scipy",
        "pandas",
        "tiktoken",
        "openai>=1.3.3",
        "tenacity",
        "jellyfish",
        "pyyaml",
    ],
)
