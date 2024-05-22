from .get_balanced_sample import get_balanced_sample
from .scale_time_amount import scale_time_amount
import pandas as pd


def base_data_preprocessor(df, important_columns):
    balanced_sample = get_balanced_sample(df, "Class")
    scaled_balanced_sample = scale_time_amount(balanced_sample)
    important_columns_balanced_scaled_sample = scaled_balanced_sample[important_columns]
    result_df = important_columns_balanced_scaled_sample
    return result_df

