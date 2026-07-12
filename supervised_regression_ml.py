import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


data = {
    "code_churn": [120, 300, 250, 500, 700, 150, 400, 650, 220, 800],
    "failed_tests": [2, 5, 4, 8, 10, 1, 6, 9, 3, 12],
    "test_coverage": [85, 75, 78, 65, 60, 90, 70, 62, 82, 55],
    "previous_incidents": [1, 2, 1, 4, 5, 0, 3, 4, 1, 6],
    "defect_count": [3, 7, 6, 12, 15, 2, 9, 14, 5, 18],
}


def calculate_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def main():
    df = pd.DataFrame(data)

    X = df[["code_churn", "failed_tests", "test_coverage", "previous_incidents"]]
    y = df["defect_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)

        results.append(
            {
                "Model": model_name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2 Score": round(r2, 2),
                "MAPE %": round(mape, 2),
            }
        )

    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()
