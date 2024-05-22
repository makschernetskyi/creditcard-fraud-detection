import pandas as pd
from sklearn.linear_model import LogisticRegression
from .grid_search_cv_parameters import logistic_regression_parameters
from .get_best_parameters_from_pipeline_for_classifier import get_best_parameters_from_pipeline_for_classifier


file_csv = "./dataset/creditcard.csv"

df = pd.read_csv(file_csv)

best_params = get_best_parameters_from_pipeline_for_classifier(LogisticRegression,  df, logistic_regression_parameters)
