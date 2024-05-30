import pandas as pd
from sklearn.svm import SVC
from .grid_search_cv_parameters import svc_parameters
from .get_best_parameters_from_pipeline_for_classifier import get_best_parameters_from_pipeline_for_classifier


file_csv = "./dataset/creditcard.csv"
df = pd.read_csv(file_csv)

best_params = get_best_parameters_from_pipeline_for_classifier(SVC, df, svc_parameters)

#plot_roc_and_scores(SVC, {'C': 0.5, 'gamma': 10, 'kernel': 'rbf'}, df)