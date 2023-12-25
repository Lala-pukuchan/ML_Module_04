import numpy as np
from ex03.logistic_loss_reg import reg_log_loss_


y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_hat = np.array([0.9, 0.79, 0.12, 0.04, 0.89, 0.93, 0.01]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

print("---Ex.1---")
# Example :
print("reg_log_loss_(y, y_hat, theta, .5):\n", reg_log_loss_(y, y_hat, theta, 0.5))
# Output:
print("expexted:\n", 0.43377043716475955)

print("\n---Ex.2---")
# Example :
print("reg_log_loss_(y, y_hat, theta, .05):\n", reg_log_loss_(y, y_hat, theta, 0.05))
# Output:
print("expexted:\n", 0.13452043716475953)

print("\n---Ex.3---")
# Example :
print("reg_log_loss_(y, y_hat, theta, .9):\n", reg_log_loss_(y, y_hat, theta, 0.9))
# Output:
print("expexted:\n", 0.6997704371647596)
