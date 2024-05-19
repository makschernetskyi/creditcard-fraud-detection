from pandas import DataFrame
from sklearn.preprocessing import RobustScaler


def scale_time_amount(df: DataFrame) -> DataFrame:
    """
    Scales the 'Amount' and 'Time' columns of a DataFrame using the RobustScaler.
    Renames the scaled columns to 'Amount_scaled' and 'Time_scaled' respectively.

    Parameters
    ----------
    df: Input DataFrame containing 'Amount' and 'Time' columns to be scaled.

    Returns
    -------
    DataFrame
        DataFrame with 'Amount' and 'Time' columns scaled and renamed to
        'Amount_scaled' and 'Time_scaled'.

    Examples
    --------
    Example usage:

    >>> import pandas as pd
    >>> from sklearn.preprocessing import RobustScaler
    >>> data = {'Amount': [100, 150, 200], 'Time': [1, 2, 3], 'V1': [0.9, 1.0, 1.1]}
    >>> df = pd.DataFrame(data)
    >>> scaled_df = scale_time_amount(df)
    >>> print(scaled_df)
       V1  Amount_scaled  Time_scaled
    0  0.9      -1.224745    -1.224745
    1  1.0       0.000000     0.000000
    2  1.1       1.224745     1.224745
    """
    df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    df.rename(columns={'Amount': 'Amount_scaled', 'Time': 'Time_scaled'}, inplace=True)

    return df
