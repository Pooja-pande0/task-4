
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the cleaned data
df = pd.read_csv("cleaned_logistics_data.csv")

# 2. Select features and target
features = [
    "traffic_congestion_level",
    "shipping_costs",
    "fuel_consumption_rate",
    "warehouse_inventory_level",
    "supplier_reliability_score",
    "historical_demand"
]

target = "lead_time_days"

# 3. Prepare the data
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train two models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8, random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"\n{name}")
    print(f"MAE: {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R2 Score: {r2:.3f}")

# 6. Compare model results
results_df = pd.DataFrame(results)
results_df.to_csv("week4_model_results.csv", index=False)
print("\nModel comparison:")
print(results_df)

# 7. Plot actual vs predicted values for the better model
best_model_name = results_df.sort_values("MAE").iloc[0]["Model"]
best_model = models[best_model_name]
predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Lead Time (days)")
plt.ylabel("Predicted Lead Time (days)")
plt.title(f"Actual vs Predicted Delivery Time: {best_model_name}")
plt.tight_layout()
plt.savefig("week4_actual_vs_predicted.png")
plt.show()

# 8. Cross-validation on training data
cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=5, scoring="neg_mean_absolute_error"
)
print("Average 5-fold CV MAE:", -cv_scores.mean())
