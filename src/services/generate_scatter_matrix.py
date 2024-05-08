import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid


def generate_scatter_matrix(path):
    df = pd.read_csv("./dataset/creditcard.csv")
    pd.plotting.scatter_matrix(df.head(), figsize=(12, 8))
    matrix_id = uuid.uuid4()
    plt.savefig(path + f'pd_scatter_matrix_{matrix_id}.png')
    plt.show()
