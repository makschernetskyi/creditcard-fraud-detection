import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .grid_search_cv_paramaters import random_forest_parameters

from .get_best_parameters_from_pipeline_for_classifier import get_best_parameters_from_pipeline_for_classifier

file_csv = "./dataset/creditcard.csv"

df = pd.read_csv(file_csv)

get_best_parameters_from_pipeline_for_classifier(RandomForestClassifier, df, random_forest_parameters)



