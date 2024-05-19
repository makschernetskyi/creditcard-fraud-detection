import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
from .get_balanced_sample import get_balanced_sample

csv_file = "./dataset/creditcard.csv"


def generate_correlation_heatmap(path, lines=None, balanced=False):
    """
        Generates a correlation heatmap from a CSV file and saves it as an image.

        Parameters
        ----------
        path : str
            The file path where the correlation heatmap image will be saved.
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

            generate_correlation_heatmap('/path/to/save/image', lines=100, balanced=True)

        This will generate a correlation heatmap from the first 100 lines of the CSV file and balance the classes.

        The output image will be saved at the specified path with a unique identifier.

        Notes
        -----
        - The CSV file is expected to have a column named 'Class' if the `balanced` parameter is set to True.
        - The generated image file will include the number of lines used (or 'ALL' if all lines are used), the balance status, and a unique ID in its filename.
        """

    df = pd.read_csv(csv_file)

    if balanced:
        df = get_balanced_sample(df, 'Class')
    elif lines is not None:
        df = df.head(int(lines))
    corr = df.corr()
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr, cmap='coolwarm_r')
    matrix_id = uuid.uuid4()
    plt.savefig(path + f'correlation_heatmap_lines_{lines if lines else "ALL"}_{"balanced" if balanced else "not_balanced"}_id_{matrix_id}.png')

