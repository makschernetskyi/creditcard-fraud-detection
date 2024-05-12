import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid


def generate_scatter_matrix(path, lines=None):
    df = pd.read_csv("./dataset/creditcard.csv")
    if lines is not None:
        df = df.head(int(lines))
    pd.plotting.scatter_matrix(df, figsize=(24, 20))
    matrix_id = uuid.uuid4()
    plt.savefig(path + f'pd_scatter_matrix_lines_{lines if lines else "ALL"}_id_{matrix_id}.png')

