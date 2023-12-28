from ex00.polynomial_model_extended import add_polynomial_features as add_polynomial_features_extended
from ex07.polynomial_model import add_polynomial_features as add_polynomial_features
import numpy as np

x = np.arange(1, 11).reshape(5, 2)
# Example 1:
print("add_polynomial_features:\n", add_polynomial_features(x, 3))
print("add_polynomial_features_extended:\n", add_polynomial_features_extended(x, 3))