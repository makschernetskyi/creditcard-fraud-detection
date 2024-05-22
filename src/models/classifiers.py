from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture

models = {
    # supervised:
    "svc": SVC(),
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "k_nearest": KNeighborsClassifier(),
    "decision_tree_classifier": DecisionTreeClassifier(),
    # unsupervised:
    "gaussian_mixture": GaussianMixture(),
    "isolation_forest": IsolationForest()
}

