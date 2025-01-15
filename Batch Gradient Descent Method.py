import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = load_iris()
X = data.data
y = data.target
y = y[y != 2]  # Only use classes 0 and 1 for binary classification
X = X[:len(y)]  # Match X and y sizes

# Split and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Batch Gradient Descent
def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    bias = 0
    losses = []

    for epoch in range(epochs):
        y_pred = np.dot(X, theta) + bias
        error = y_pred - y
        total_loss = np.sum(error ** 2)

        d_theta = np.dot(X.T, error)
        d_bias = np.sum(error)

        theta -= lr * d_theta / m
        bias -= lr * d_bias / m

        losses.append(total_loss / m)

    return theta, bias, losses

# Train and make predictions
theta, bias, losses = batch_gradient_descent(X_train, y_train)

# Make predictions
y_pred = np.dot(X_test, theta) + bias

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# TSS and RSS
tss = np.sum((y_test - np.mean(y_test)) ** 2)
rss = np.sum((y_test - y_pred) ** 2)

# Print results
print("Batch Gradient Descent Results:")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, TSS: {tss:.4f}, RSS: {rss:.4f}")

# Visualizations
# Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(losses, label="Loss (MSE)", color="purple")
plt.title("Batch Gradient Descent: Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="orange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Batch Gradient Descent: Predicted vs Actual")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Bar plot for all metrics including TSS and RSS
metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'TSS', 'RSS']
values = [mae, mse, rmse, r2, tss, rss]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
plt.title("Batch Gradient Descent: Performance Metrics")
plt.ylabel("Value")
plt.show()
