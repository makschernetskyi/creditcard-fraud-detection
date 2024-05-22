import numpy as np

# PARAMETERS FOR GRID SEARCH

svc_parameters = {
    'C': [0.2, 0.4, 0.6, 0.8, 1.],  # Lower values-stronger regularization
    'kernel': ['rbf', 'poly', 'sigmoid', 'linear']  # Kernel type
}

logistic_regression_parameters = {
    'penalty': ['l1', 'l2'],  # Norm used in the penalization
    'C': np.logspace(-4, 4, 20)  # Inverse of regularization strength
}

random_forest_parameters = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

k_nearest_parameters = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'p': [1, 2]  # Power parameter for Minkowski distance
}

decision_tree_classifier_parameters = {
    "criterion": ["gini", "entropy"],  # Criteria for quality of split
    "max_depth": np.arange(2, 4, 1),  # Maximum depth of the tree
    "min_samples_leaf": np.arange(5, 7, 1)  # Minimum number of samples required to be at a leaf node
}


gaussian_mixture_parameters = {
    'n_components': [2, 3, 4, 5],           # Number of mixture components
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Type of covariance parameters to use
    'max_iter': [50, 100, 200]              # Maximum number of EM iterations to perform
}


isolation_forest_parameters = {
    'n_estimators': [50, 100, 200],         # Number of base estimators in the ensemble
    'max_samples': ['auto', 100, 200],      # Number of samples to draw from X to train each base estimator
    'contamination': [0.1, 0.2, 0.3]        # The proportion of outliers in the data set
}