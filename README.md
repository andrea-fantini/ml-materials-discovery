ml-materials-discovery
==============================

Accelerate material discovery with machine learning. In this project I use a very small dataset of electrolyte compositions and their lab measured properties to build a model that aims at understanding the relationship between composistion and property. Then I develop a generative model to identify new compositions that maximize the desired property. These compositions are sent to the lab for anlaysis and the process is repeteated. This enables a faster discovery of new materials as opposed to the traditional trial and error method.


Project Organization
------------

    │
    ├── data/               <- The original, immutable data dump. 
    │
    ├── figures/            <- Figures saved by scripts or notebooks.
    │
    ├── src/      <- Python module with source code of this project.
    │
    ├── environment.yml     <- conda virtual environment definition file.
    │
    ├── Materials Discovery.ipynb     <- Jupyter Notebook that contains the exploratory data analysis, modeling, and new material generation.
    │
    ├── LICENSE
    │
    └── README.md           <- The top-level README for developers using this project.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</p>


Set up
------------

Install the virtual environment with conda and activate it:

```bash
$ conda env create -f environment.yml
$ conda activate example-project 
```

Install `src` in the virtual environment:

```bash
$ pip install --editable .
```
