import numpy as np

# PARAMETERS FOR GRID SEARCH

svc_parameters = {
    'C': [0.01, 0.1, 0.5, 1., 10, 50],  # Lower values - stronger regularization
    'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],  # Kernel type
    'gamma': ['scale', 'auto', 0.01, 0.1, 1., 10, 500]  # Higher values - more fitting
}

logistic_regression_parameters = {
    'penalty': ['l2'],  # Norm used in the penalization
    'C': np.logspace(-4, 4, 20)  # Inverse of regularization strength
}

random_forest_parameters = {
    'n_estimators': [50, 100, 200, 500],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_leaf_nodes': [10, 16, 20, 30],  # Maximum number of leaf nodes, fixed at 16
}

k_neighbors_parameters = {
    'n_neighbors': [1, 3, 5, 7, 9],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'p': [1, 2, 3]  # Power parameter for Minkowski distance
}

decision_tree_classifier_parameters = {
    "criterion": ["gini", "entropy"],  # Criteria for quality of split
    "max_depth": [3, 5, 7, 10, 20],  # Maximum depth of the tree
    "min_samples_leaf": [1, 2, 3, 5, 10],  # Minimum number of samples required to be at a leaf node
    "max_leaf_nodes": [2, 5, 10, 20, 100, None],  # Different values for maximum number of leaf nodes
}

gaussian_mixture_parameters = {
    'n_components': [1, 3, 5, 7],   # Number of mixture components
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # Type of covariance parameters to use
    'tol': [1e-3, 1e-4, 1e-5],                           # Convergence threshold
    'reg_covar': [1e-6, 1e-5, 1e-4],                     # Regularization on covariance to avoid singular matrices
    'max_iter': [100, 200, 300, 400]                # Maximum number of iterations
}


isolation_forest_parameters = {
    'n_estimators': [50, 100, 200],         # Number of base estimators in the ensemble
    'max_samples': ['auto', 100, 200],      # Number of samples to draw from X to train each base estimator
    'contamination': [0.1, 0.2, 0.3]        # The proportion of outliers in the data set
}
