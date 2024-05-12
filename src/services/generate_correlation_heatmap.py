import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uuid


def generate_correlation_heatmap(path, lines=None):
    df = pd.read_csv("./dataset/creditcard.csv")
    if lines is not None:
        df = df.head(int(lines))
    corr = df.corr()
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr, cmap='coolwarm_r')
    matrix_id = uuid.uuid4()
    plt.savefig(path + f'correlation_heatmap_lines_{lines if lines else "ALL"}_id_{matrix_id}.png')

