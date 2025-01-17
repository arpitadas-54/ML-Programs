import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

iris=pd.read_csv('IRIS-DATA.csv')
print(iris.head())

X=iris[['petal_length','petal_width','sepal_width']]
y=iris['sepal_length']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

tss = np.sum((y_test - np.mean(y_test))**2)

rss = np.sum((y_test - y_pred)**2)

ess = tss - rss

mae = mean_absolute_error(y_test, y_pred)

print(f"TSS: {tss}")
print(f"RSS: {rss}")
print(f"ESS: {ess}")
print(f"MAE: {mae}")


y_pred=model.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_squared_error, r2_score
mse= mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
print("plot saved to iris scatterplot.png")

metrics = ['TSS', 'RSS', 'ESS']
values = [tss, rss, ess]

plt.bar(metrics, values, color=['blue', 'red', 'green'])
plt.title('TSS, RSS, and ESS for Multiple Linear Regression')
plt.ylabel('Values')
plt.show()

plt.bar('MAE', mae, color='purple')
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('Value')
plt.show()


import joblib
joblib.dump(model,"Linear_regression_model.pkl")
joblib.load("linear_regression_model.pkl")

results=pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results.to_csv("predictions.csv", index=False)
print("Predictions saved to prdiction.csv")

with open("metrics.txt", "w") as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R-Squared: {r2}\n")
print("Metrics saved to metrics.txt")