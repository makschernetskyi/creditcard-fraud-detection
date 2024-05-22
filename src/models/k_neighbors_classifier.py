import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from .grid_search_cv_parameters import k_neighbors_parameters
from .get_best_parameters_from_pipeline_for_classifier import get_best_parameters_from_pipeline_for_classifier
from .functions import scores

file_csv = "./dataset/creditcard.csv"

df = pd.read_csv(file_csv)

best_params = get_best_parameters_from_pipeline_for_classifier(KNeighborsClassifier,  df, k_neighbors_parameters)

#scores = scores(KNeighborsClassifier, best_params, df) ?????????
