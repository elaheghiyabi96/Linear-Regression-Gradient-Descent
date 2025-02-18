# Linear-Regression-Gradient-Descent
A simple implementation of Linear Regression using NumPy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
# Set seed for reproducibility
np.random.seed(0)
# Generate synthetic data
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# Plot the data
plt.scatter(X, Y, color='orange', label='Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()

# Initialize parameters
w = 0.1  # Weight (slope)
b = 0.1  # Bias (intercept)
learning_rate = 0.01
iterations = 1000
# Gradient descent
for i in range(iterations):
    predictions = w * X + b
    # Compute gradients
    dw = (1 / len(X)) * np.sum((predictions - Y) * X)  # Gradient with respect to w
    db = (1 / len(X)) * np.sum(predictions - Y)        # Gradient with respect to b
    
    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
  
    # Compute cost function
    m = len(X)
    predictions = w * X + b
    cost = (1 / (2 * m)) * np.sum((predictions - Y) ** 2)  # Mean Squared Error (MSE)
    
    # Print cost every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Cost {cost}")
    
# Compute final linear regression line
y = w * X + b

# Plot regression line
plt.plot(X, y, label='Fitted Line', color='b')
plt.legend()
plt.show()
