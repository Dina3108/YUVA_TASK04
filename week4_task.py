# Week 4 - Training and Checking a Linear Regression Model
# The target is a number, so this is a regression task.

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

output_folder = Path("week4_outputs")
output_folder.mkdir(exist_ok=True)

data = load_diabetes(as_frame=True).frame
features = data.drop(columns="target")
target = data["target"]

# Keep some data aside so the model is tested on examples it did not train on.
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r_squared = r2_score(y_test, predictions)

print("Model results on the test data")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R-squared:", round(r_squared, 3))

# A point close to the diagonal means the prediction was close to the actual value.
plt.figure(figsize=(6, 5))
plt.scatter(y_test, predictions, alpha=0.7)
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], "--")
plt.xlabel("Actual target")
plt.ylabel("Predicted target")
plt.title("Actual Values and Model Predictions")
plt.tight_layout()
plt.savefig(output_folder / "actual_vs_predicted.png", dpi=150)
plt.close()

# Residuals are the actual values minus the predictions.
errors = y_test - predictions
plt.figure(figsize=(6, 5))
plt.scatter(predictions, errors, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted target")
plt.ylabel("Prediction error")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(output_folder / "residual_plot.png", dpi=150)
plt.close()

print("Charts saved in", output_folder)
