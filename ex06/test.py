from ridge import MyRidge
import numpy as np

X = np.array([[1.0, 1.0, 2.0, 3.0], [5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]])
Y = np.array([[23.0], [48.0], [218.0]])
mylr = MyRidge(np.array([[1.0], [1.0], [1.0], [1.0], [1]]), lambda_=0.0)

print("---------------------------------------------------")
# Example 0:
y_hat = mylr.predict_(X)
print("Predicted y_hat:", y_hat)
print("Expected y_hat (Example 0):", "[[8.], [48.], [323.]]")

print("---------------------------------------------------")
# Example 1:
loss_elem = mylr.loss_elem_(Y, y_hat)
print("Loss element-wise:", loss_elem)
print("Expected loss element-wise (Example 1):", "[[225.], [0.], [11025.]]")

print("---------------------------------------------------")
# Example 2:
loss = mylr.loss_(Y, y_hat)
print("Total loss:", loss)
print("Expected total loss (Example 2):", "1875.0")

print("---------------------------------------------------")
# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print("Fitted thetas:", mylr.theta)
print(
    "Expected thetas (Example 3):",
    "[[1.8188..], [2.767..], [-3.74..], [1.392..], [1.7..]]",
)

print("---------------------------------------------------")
# Example 4:
y_hat = mylr.predict_(X)
print("Predicted y_hat after fit:", y_hat)
print("Expected y_hat (Example 4):", "[[23.417..], [47.489..], [218.065...]]")

print("---------------------------------------------------")
# Example 5:
loss_elem = mylr.loss_elem_(Y, y_hat)
print("Loss element-wise after fit:", loss_elem)
print("Expected loss element-wise (Example 5):", "[[0.174..], [0.260..], [0.004..]]")

print("---------------------------------------------------")
# Example 6:
loss = mylr.loss_(Y, y_hat)
print("Total loss after fit:", loss)
print("Expected total loss (Example 6):", "0.0732..")

mylr.set_params_(lambda_=2.0)
print("Parameters after setting lambda_ to 2.0:", mylr.get_params_())
