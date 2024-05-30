import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, classification_report, RocCurveDisplay, auc
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, RepeatedStratifiedKFold, \
    cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from services import base_data_preprocessor


def plot_roc_and_scores(model, best_parameters, df):
    """
    Preprocesses the data, trains a model, evaluates it using cross-validation,
    plots the ROC curve (saves the plot), and prints the classification report.

    Parameters:
    model (class): The machine learning model class to be instantiated (e.g., LogisticRegression).
    best_parameters (dict): A dictionary of the best hyperparameters for the model.
    df (pandas.DataFrame): The input DataFrame containing the data to be processed and used for training/testing.

    The function follows these steps:
    1. Preprocesses the data using the base_data_preprocessor function.
    2. Splits the preprocessed data into training and testing sets.
    3. Instantiates and trains the model with the provided best parameters on the training set.
    4. Evaluates the model using cross-validation and prints the average ROC AUC score.
    5. Predicts probabilities using cross-validation on the training set.
    6. Computes the ROC curve and AUC score.
    7. Plots and saves the ROC curve.
    8. Prints the classification report for the test set predictions.

    Returns:
    None
    """
    preprocessed_df = base_data_preprocessor(df, ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12',
                                                  'V14', 'V16', 'V17', 'V18', 'Class'])

    X = preprocessed_df[
        ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']]
    y = preprocessed_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # if SVC is used, set 'probability' parameter to True in order to be able to perform cross_val_predict
    if model == SVC:
        best_parameters['probability'] = True
    classifier = model(**best_parameters)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)

    kf = StratifiedKFold(5)
    print("Cross Validation Score : ",
          '{0:.2%}'.format(cross_val_score(classifier, X_train, y_train, cv=kf, scoring='roc_auc').mean()))

    y_pred = cross_val_predict(classifier, X_train, y_train, method='predict_proba',
                               cv=kf)
    y_pred = y_pred[:, 1]

    fpr, tpr, thresholds = roc_curve(y_train, y_pred)

    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=repr(model))
    display.plot()
    plt.savefig('../plots/' + f'ROC_AUC_{type(classifier).__name__}.png')
    plt.show()
    print(classification_report(y_test, prediction))


# run function for a model
file_csv = "../dataset/creditcard.csv"
df = pd.read_csv(file_csv)

plot_roc_and_scores(RandomForestClassifier, {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}, df)
