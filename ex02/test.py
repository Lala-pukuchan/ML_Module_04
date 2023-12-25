import numpy as np
from ex02.linear_loss_reg import reg_loss_

y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

print("---Ex.1---")
# Example :
print("reg_loss_(y, y_hat, theta, .5):\n", reg_loss_(y, y_hat, theta, 0.5))
# Output:
print("expexted:\n", 0.8503571428571429)

print("\n---Ex.2---")
print("reg_loss_(y, y_hat, theta, .05):\n", reg_loss_(y, y_hat, theta, 0.05))
# Output:
print("expexted:\n", 0.5511071428571429)


print("\n---Ex.3---")
# Example :
print("reg_loss_(y, y_hat, theta, .9):\n", reg_loss_(y, y_hat, theta, 0.9))
# Output:
print("expexted:\n", 1.116357142857143)
