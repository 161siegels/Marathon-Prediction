import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def tune(x_train, y_train, x_valid, y_valid, cat_features):
    def objective(trial):
        param = {
            "eval_metric": "RMSE",
            "loss_function": "RMSE",
            "learning_rate": trial.suggest_discrete_uniform(
                "learning_rate", 0.001, 0.02, 0.001
            ),
            "depth": trial.suggest_int("depth", 1, 5),
            "l2_leaf_reg": trial.suggest_discrete_uniform("l2_leaf_reg", 1.0, 5.5, 0.5),
            " ": trial.suggest_categorical(
                "min_child_samples", [1, 4, 8],
            ),
        }

        gbm = CatBoostRegressor(**param, cat_features=cat_features)

        gbm.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=1,
            early_stopping_rounds=100,
        )

        loss = mean_squared_error(y_valid, gbm.predict(x_valid.copy()))
        return loss

    study = optuna.create_study(study_name=f"catboost")
    study.optimize(objective, n_trials=50, n_jobs=4, timeout=24000)
    return study.best_params
