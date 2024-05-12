from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, RocCurveDisplay, auc
from sklearn.model_selection import GridSearchCV


def grid_search_best_params(model, parameters, X_train, y_train):
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    grid = GridSearchCV(model, param_grid=parameters, cv=kf,
                        scoring='recall').fit(X_train, y_train)
    print(f"Best parameters for {repr(model)}:", grid.best_params_)
    print(f"Best score for {repr(model)}:", grid.best_score_)
    return grid.best_params_


def scores(model, best_parameters, X_train, y_train, x_test, y_test):
    model(best_parameters).fit(X_train, y_train)
    prediction = model.predict(x_test)
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)  #
    print("Cross Validation Score : ",
          '{0:.2%}'.format(cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc').mean()))
    fpr, tpr, thresholds = roc_curve(y_test, prediction)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=repr(model))
    display.plot()
    plt.show()
    print(classification_report(y_test, prediction))
