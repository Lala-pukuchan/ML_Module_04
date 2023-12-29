from my_logistic_regression import MyLogisticRegression as mylogr
import numpy as np

theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print("\n===TEST FOR LAMBDA===\n")
# Example 1:
model1 = mylogr(theta, lambda_=5.0)
print("Example 1:")
print("Actual penalty:", model1.penalty, " | Expected: 'l2'")
print("Actual lambda_:", model1.lambda_, " | Expected: 5.0")

# Example 2:
model2 = mylogr(theta, penalty=None)
print("\nExample 2:")
print("Actual penalty:", model2.penalty, " | Expected: None")
print("Actual lambda_:", model2.lambda_, " | Expected: 0.0")

# Example 3:
model3 = mylogr(theta, penalty=None, lambda_=2.0)
print("\nExample 3:")
print("Actual penalty:", model3.penalty, " | Expected: None")
print("Actual lambda_:", model3.lambda_, " | Expected: 0.0")

print("\n===TEST FOR GRAD===")
x = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
model4 = mylogr(theta, penalty=None, lambda_=2.0)


print("\n---Ex.1.2---")
# Example 1.2:
model4.set_params_(lambda_=1)
print("vec_reg_logistic_grad(y, x, theta, 1):\n", model4.vec_reg_logistic_grad(y, x))
# Output:
print(
    "expexted:\n",
    np.array(
        [[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]]
    ),
)

print("\n---Ex.2.2---")
# Example 2.2:
model4.set_params_(lambda_=0.5)
print(
    "vec_reg_logistic_grad(y, x, theta, 0.5):\n",
    model4.vec_reg_logistic_grad(y, x),
)
# Output:
print(
    "expexted:\n",
    np.array(
        [[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]]
    ),
)

# Example 3.2:
model4.set_params_(lambda_=0.0)
print(
    "vec_reg_logistic_grad(y, x, theta, 0.0)\n", model4.vec_reg_logistic_grad(y, x)
)
# Output:
print(
    "expexted:\n",
    np.array(
        [[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]]
    ),
)


print("\n===TEST FOR LOSS===\n")
y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_hat = np.array([0.9, 0.79, 0.12, 0.04, 0.89, 0.93, 0.01]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
model5 = mylogr(theta, penalty=None, lambda_=2.0)

print("---Ex.1---")
# Example :
model5.set_params_(lambda_=0.5)
print("reg_log_loss_(y, y_hat, theta, .5):\n", model5.loss_(y, y_hat))
# Output:
print("expexted:\n", 0.43377043716475955)

print("\n---Ex.2---")
model5.set_params_(lambda_=0.05)
# Example :
print("reg_log_loss_(y, y_hat, theta, .05):\n", model5.loss_(y, y_hat))
# Output:
print("expexted:\n", 0.13452043716475953)

print("\n---Ex.3---")
model5.set_params_(lambda_=0.9)
# Example :
print("reg_log_loss_(y, y_hat, theta, .9):\n", model5.loss_(y, y_hat))
# Output:
print("expexted:\n", 0.6997704371647596)
