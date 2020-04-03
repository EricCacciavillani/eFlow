# eflow
I designed this project to help make my life easier for my data science projects. It uses a combination of generating code in cell blocks (Check my testing/templates for example notebooks) and well designed objects for analysis and modeling. Currently it is far from done; (studying for GRE and have a full time job in my defense: ) ).

This project was mainly designed for and in Jupyter-Lab. I don't know how well it works in Jupyter-Notebook.

# Documentation
Can be found by clicking on docs/build/html/index.html in eflow using sphinx or go to https://eflow.readthedocs.io/en/latest/index.html.

# Installation
```bash
$ pip install eflow
```
# Other commands/steps to make the project work properly
Ensuring the widgets work properly
```bash
$ jupyterlab nbextension enable --py widgetsnbextension --sys-prefix
$ jupyter nbextension enable --py widgetsnbextension --sys-prefix
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
$ jupyter lab clean
$ jupyter lab build
```
Getting natural language datasets setup. Start by opening up a Python Repl.

```bash
$ python
>>> nltk.download('wordnet')
>>> nltk.download('words')
>>> nltk.download('punkt')
```

# Project Requirements
* Python >= 3.7 
* Latest version's of the following packages:
    * jupyterlab
    * numpy
    * pandas
    * missingno
    * matplotlib
    * seaborn
    * sklearn
    * kneed
    * Pillow
    * tqdm
    * python-dateutil
    * scikit-plot
    * scipy
    * ipywidgets
    * ipython_blocking
    * nltk
