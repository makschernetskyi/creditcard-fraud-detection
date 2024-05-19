import pandas as pd
import matplotlib.pyplot as plt
import uuid
from .get_balanced_sample import get_balanced_sample

csv_file = "./dataset/creditcard.csv"


def generate_scatter_matrix(path, lines=None, balanced=False):
    """
    Generates a scatter matrix from a CSV file and saves it as an image.

    Parameters
    ----------
    path : str
        The file path where the scatter matrix image will be saved.
    lines : int, optional
        The number of lines to read from the CSV file. If None, all lines will be read. Default is None.
    balanced : bool, optional
        Whether to balance the dataset by downsampling each group to the size of the smallest group. Default is False.

    Returns
    -------
    None

    Examples
    --------
    Example usage:

        generate_scatter_matrix('/path/to/save/image', lines=100, balanced=True)

    This will generate a scatter matrix from the first 100 lines of the CSV file and balance the classes.

    The output image will be saved at the specified path with a unique identifier.

    Notes
    -----
    - The CSV file is expected to have a column named 'Class' if the `balanced` parameter is set to True.
    - The generated image file will include the number of lines used (or 'ALL' if all lines are used) and a unique ID in its filename.
    """

    df = pd.read_csv(csv_file)
    if balanced:
        df = get_balanced_sample(df, 'Class')
    if lines is not None:
        df = df.head(int(lines))
    pd.plotting.scatter_matrix(df, figsize=(24, 20))
    matrix_id = uuid.uuid4()
    plt.savefig(path + f'pd_scatter_matrix_lines_{lines if lines else "ALL"}_id_{matrix_id}.png')

