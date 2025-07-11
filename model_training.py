import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from math import sqrt

# Load data
df = pd.read_csv("cleaned_medical_insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = None
best_score = float('inf')
best_run_id = None

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = sqrt(mean_squared_error(y_test, preds))
        mlflow.log_param("model_name", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mean_absolute_error(y_test, preds))
        mlflow.log_metric("r2", r2_score(y_test, preds))
        mlflow.sklearn.log_model(model, "model")

        if rmse < best_score:
            best_model = model
            best_score = rmse
            best_run_id = run.info.run_id

print(f"✅ Best model saved in run ID: {best_run_id}")