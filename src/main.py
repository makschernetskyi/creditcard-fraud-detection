import pandas as pd
from commands import define_commands
from models.functions import scores
from sklearn.tree import DecisionTreeClassifier


def main():
    define_commands()
    file_csv = "./dataset/creditcard.csv"
    df = pd.read_csv(file_csv)
    scores(DecisionTreeClassifier, {'criterion': 'gini', 'max_depth': 3, 'max_leaf_nodes': 5, 'min_samples_leaf': 10}, df)


if __name__ == "__main__":
    main()
