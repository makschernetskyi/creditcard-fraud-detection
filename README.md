# creditcard-fraud-detection

ML credit card detection project

## Description
Project analyzes dataset "credit card fraud detection" using a variety of ML methods and techniques
to process it and output something.


## dataset source
[dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
This dataset should be in the folder src/dataset, which is included to gitignore.
Therefore, when cloning the repository, please fetch the required data.

## Usage

the project is functioning as a list of commands executable as

`python main.py <command> <options>`

example `python main.py generatescattermatrix --lines 1024`

### list of available commands

1. generatescattermatrix - generates pandas scatter matrix for the dataset.
    - options:
      - --path - defines path to save the plot
      - --lines - defines how many lines of the original data will be used for the plot
2. generatecorrelationheatmap - generates pandas correlation heatmap for tha dataset.
    - options:
      - --path - defines path to save the plot
      - --lines - defines how many lines of the original data will be used for the plot