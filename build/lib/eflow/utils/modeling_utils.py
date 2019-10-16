from sklearn.model_selection import GridSearchCV


def optimize_model_grid(model,
                        X_train,
                        y_train,
                        param_grid,
                        scoring="f1_macro",
                        cv=5,
                        verbose=0,
                        n_jobs=1):
    """
    model:
        Machine learning model to fit across a cross fold with a cross

    X_train:
        Feature matrix.

    y_train:
        Target vector.

    param_grid:
        Dictionary with parameters names.

    scoring:
        String value to determine the metric to evaluate the best model.
        Link to all strings: http://tinyurl.com/y22f3m5k

    cv:
        Cross-validation strategy for 'training'.

    Returns/Desc:
        Finds the best parameters for a grid; returns the model and parameters.
    """

    # Instantiate the GridSearchCV object
    model_cv = GridSearchCV(model,
                            param_grid,
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            scoring=scoring)

    # Fit it to the data
    model_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print("Tuned Parameters: {}".format(model_cv.best_params_))
    print("Best score on trained data was {0:4f}".format(model_cv.best_score_))

    # Return model and parameters
    return model_cv.best_estimator_, model_cv.best_params_

