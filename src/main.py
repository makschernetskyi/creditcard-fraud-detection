import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from commands import define_commands
from models.functions import scores


def main():
    define_commands()
    file_csv = "./dataset/creditcard.csv"
    df = pd.read_csv(file_csv)
    scores(KNeighborsClassifier, {'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}, df)


if __name__ == "__main__":
    main()
