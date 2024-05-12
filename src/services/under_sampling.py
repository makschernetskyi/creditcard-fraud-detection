import pandas as pd


def scale_time_amount(save_path, read_path="./dataset/creditcard_scaled_time_amount.csv"):
    df = pd.read_csv(read_path)
    df = df.sample(frac=1)  # shuffle rows
    df_fraud = df.loc[df['Class'] == 1]  # 492 rows
    df_non_fraud = df.loc[df['Class'] == 0][:492]  # take 492 rows for balancing
    df_balanced = pd.concat([df_fraud, df_non_fraud])
    df_balanced = df_balanced.sample(frac=1, random_state=42)  # shuffle
    df_balanced.to_csv(save_path + "creditcard_balanced.csv", index=False)
