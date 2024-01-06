# Predicting Nitrogen Level in Potato Plant.

# Goal 
This project aims to predict the nitrogen that have been added to the potato, based on hyperspectral images. In developing this model, we aim to apply what we have learned in this course 02476 Machine Learning Operations.


# DATA
The images consist of 4 set of images taken in different spectrums, NIR, green, red and red edge. The data is provided by the company Aerial Tools. 160*4 images have been taken in the 4 spectrums. 25 percent of the images will be as the test set. The rest of the data will be used to train the neural network on. The images have the size of 160 times 106 pixels. Different image augmentations (flipping, cropping) will be used to try to achieve better performance and cropping might be an improvement if a fraction of an image contains enough information to predict nitrogen content.


# Method
A neural network will be developed to do regression on the images. The neural network will develop in Pytorch. We expect to use a CNN. The model’s performance will be evaluated based on the r2, RMSE and MAE.

## Work plan structure of project
Here is a google drive link showing how we structured and delegated tasks.
https://docs.google.com/document/d/1lLLNnOMxulvgJ_XtQBqTcpUVArmR-BCf2veG_gLhQXo/edit?usp=sharing






# src

ml-ops project

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
