import pandas as pd
from sklearn.preprocessing import RobustScaler


# use robust scaler (less prone to outliers) for outline data - time + amount


def scale_time_amount(path):
    df = pd.read_csv("./dataset/creditcard.csv")
    df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df.rename(columns={'Amount': 'Amount_scaled', 'Time': 'Time_scaled'}, inplace=True)
    df.to_csv(path + "creditcard_scaled_time_amount.csv", index=False)
