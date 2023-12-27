import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ex06.ridge import MyRidge as MyR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features


# Load data
data = pd.read_csv("ex07/space_avocado.csv")
x = np.array(data[["weight", "prod_distance", "time_delivery"]])
y = np.array(data["target"]).reshape(-1, 1)

# Split data
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

# Scale data
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())

# Store performance of models
models_performance = {}

# Train and evaluate models with polynomial features up to degree 4
for degree in range(1, 5):
    # Create polynomial features
    x_train_poly = add_polynomial_features(x_train, degree)
    x_test_poly = add_polynomial_features(x_test, degree)

    # Initialize the model
    theta = np.zeros((x_train_poly.shape[1] + 1, 1))
    alpha = 1e-2
    max_iter = 1000000
    model = MyR(theta, alpha, max_iter)

    # Fit the model
    model.fit_(x_train_poly, y_train)

    # Evaluate the model
    train_mse = model.mse_(y_train, model.predict_(x_train_poly))
    y_pred = model.predict_(x_test_poly)
    test_mse = model.mse_(y_test, y_pred)

    # debug
    print("--- degree", degree, "---")
    print("test_mse :", test_mse)

    # Store model and performance
    models_performance[degree] = {
        "model": model,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "x_test_poly": x_test_poly,
        "y_test": y_test,
    }


# Save the models to a file
with open("ex10/models.pkl", "wb") as file:
    pickle.dump(models_performance, file)


# Plotting MSE for each model
degrees = list(models_performance.keys())
test_mses = [models_performance[d]["test_mse"] for d in degrees]
train_mses = [models_performance[d]["train_mse"] for d in degrees]
plt.figure()
plt.plot(degrees, test_mses, marker="o", linestyle="-", color="b", label="Test MSE")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Model Evaluation - Test MSE vs Polynomial Degree")
plt.legend()
plt.savefig("ex10/result_ex10_figure-1.png")
plt.close()
