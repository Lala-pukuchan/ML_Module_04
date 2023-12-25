import numpy as np
from ex01.l2_reg import iterative_l2
from ex01.l2_reg import l2

print("---Ex.1---")
x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
# Example 1:
print("iterative_l2:\n", iterative_l2(x))
# # Output:
print("expexted:\n", 911.0)

print("\n---Ex.2---")
# Example 2:
print("l2:\n", l2(x))
# # Output:
print("expexted:\n", 911.0)

print("\n---Ex.3---")
y = np.array([3, 0.5, -6]).reshape((-1, 1))
# Example 3:
print("iterative_l2:\n", iterative_l2(y))
# Output:
print("expexted:\n", 36.25)

print("\n---Ex.4---")
# Example 4:
print("l2:\n", l2(y))
# Output:
print("expexted:\n", 36.25)
