from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, RocCurveDisplay, auc
from sklearn.model_selection import GridSearchCV

from services import base_data_preprocessor


def grid_search_best_params(model, parameters, X_train, y_train):
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    grid = GridSearchCV(model, param_grid=parameters, cv=kf,
                        scoring='recall')
    grid.fit(X_train, y_train)
    print(f"Best parameters for {repr(model)}:", grid.best_params_)
    print(f"Best score for {repr(model)}:", grid.best_score_)
    return grid.best_params_


def scores(model, best_parameters, df):
    preprocessed_df = base_data_preprocessor(df, ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12',
                                                  'V14', 'V16', 'V17', 'V18', 'Class'])

    X = preprocessed_df[
        ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']]
    y = preprocessed_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model(best_parameters).fit(X_train, y_train)
    prediction = model.predict(X_test)
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)  #
    print("Cross Validation Score : ",
          '{0:.2%}'.format(cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc').mean()))
    fpr, tpr, thresholds = roc_curve(y_test, prediction)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=repr(model))
    display.plot()
    plt.show()
    print(classification_report(y_test, prediction))
