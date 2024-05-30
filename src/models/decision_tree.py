import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .grid_search_cv_parameters import decision_tree_classifier_parameters
from .get_best_parameters_from_pipeline_for_classifier import get_best_parameters_from_pipeline_for_classifier
from .functions import scores


file_csv = "./dataset/creditcard.csv"
df = pd.read_csv(file_csv)

best_params = get_best_parameters_from_pipeline_for_classifier(DecisionTreeClassifier, df,
                                                               decision_tree_classifier_parameters)

scores(DecisionTreeClassifier, {'criterion': 'gini', 'max_depth': 3, 'max_leaf_nodes': 5, 'min_samples_leaf': 10}, df)