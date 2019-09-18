from sklearn.model_selection import GridSearchCV


def optimize_model_grid(model,
                        X_train,
                        y_train,
                        param_grid,
                        cv=10):
    """
        Finds the best parameters for a grid; returns the model and parameters.
    """
    # Instantiate the GridSearchCV object
    model_cv = GridSearchCV(model,
                            param_grid,
                            cv=cv,
                            n_jobs=-1)

    # Fit it to the data
    model_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print("Tuned Parameters: {}".format(model_cv.best_params_))
    print("Best score on trained data was {0:4f}".format(model_cv.best_score_))

    # Return model and parameters
    return model_cv.best_estimator_, model_cv.best_params_

