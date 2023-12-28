import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Load dataset
data = pd.read_csv("ex07/space_avocado.csv")
x = np.array(data[["weight", "prod_distance", "time_delivery"]])
y = np.array(data["target"]).reshape(-1, 1)

# Load models and scores from the file
with open("ex07/models.pkl", "rb") as file:
    models = pickle.load(file)

# Identify the best degree
best_mse = float("inf")
best_degree = None
for degree in models:
    for lambda_, model_info in models[degree].items():
        if model_info["mse"] < best_mse:
            best_mse = model_info["mse"]
            best_degree = degree

# Print the best degree and MSE
print(f"Best degree: {best_degree}, Best MSE: {best_mse}")

# Get models of the best degree with different lambda values
lambda_models = models[best_degree]

# Create subplots for each lambda
fig, axs = plt.subplots(1, len(lambda_models), figsize=(15, 5), subplot_kw={'projection': '3d'})

for i, (lambda_, model_info) in enumerate(lambda_models.items()):
    model = model_info["model"]
    x_test_poly = model_info["x_test_poly"]
    y_test = model_info["y_test"]
    
    # Predict using the model
    y_pred = model.predict_(x_test_poly)

    # Scatter plot for actual vs predicted prices
    axs[i].scatter(x_test_poly[:, 0], x_test_poly[:, 1], y_test[:, 0], c="b", label="Actual Price")
    axs[i].scatter(x_test_poly[:, 0], x_test_poly[:, 1], y_pred[:, 0], c="r", label="Predicted Price")
    axs[i].set_title(f"Lambda {lambda_}")
    axs[i].set_xlabel("Weight")
    axs[i].set_ylabel("Production Distance")
    axs[i].set_zlabel("Price")
    axs[i].legend()

plt.suptitle("Price Prediction for Different Lambda Values")
plt.tight_layout()
plt.savefig("results/ex07/best_degree_prediction.png")
