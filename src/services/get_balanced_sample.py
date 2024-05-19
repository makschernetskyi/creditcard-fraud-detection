from pandas import DataFrame


def get_balanced_sample(df: DataFrame, key_column: str) -> DataFrame:
    """
    Downsamples each group in the DataFrame to the size of the smallest group.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the data to be balanced.
    key_column : str
        The name of the column to group by.

    Returns
    -------
    DataFrame
        A DataFrame with balanced groups, each containing the same number of samples.

    Examples
    --------
    Example usage:

        data = {
            'Feature': [10, 20, 30, 40, 50, 60, 70, 80],
            'Class': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        }
        df = pd.DataFrame(data)
        balanced_df = get_balanced_sample(df, 'Class')
        print(balanced_df)

    The output will be:

       Feature Class
    0       10     A
    1       30     A
    2       50     A
    3       20     B
    4       40     B
    5       60     B
    """
    grouped_df = df.groupby(key_column)
    return grouped_df.sample(n=grouped_df.size().min()).reset_index(drop=True)





