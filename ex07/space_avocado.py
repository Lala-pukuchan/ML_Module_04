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

# print best model
print("best degree:", best_degree, "best_mse:", best_mse)

# Get models of the best degree with different lambda values
lambda_models = models[best_degree]

# Feature pairs and labels
feature_pairs = [(0, 1), (0, 2), (1, 2)]
feature_labels = ["Weight", "Production Distance", "Time Delivery"]
dot_size = 5

# Iterate over each pair of features
for f1, f2 in feature_pairs:
    # Create a new figure for each pair
    fig, axs = plt.subplots(
        1, len(lambda_models), figsize=(15, 5), subplot_kw={"projection": "3d"}
    )

    for i, (lambda_, model_info) in enumerate(lambda_models.items()):
        model = model_info["model"]
        x_test_poly = model_info["x_test_poly"]
        y_test = model_info["y_test"]
        y_pred = model.predict_(x_test_poly)

        # Plot for each lambda
        axs[i].scatter(
            x_test_poly[:, f1],
            x_test_poly[:, f2],
            y_test[:, 0],
            c="b",
            label="Actual Price",
            s=dot_size,
        )
        axs[i].scatter(
            x_test_poly[:, f1],
            x_test_poly[:, f2],
            y_pred[:, 0],
            c="r",
            label="Predicted Price",
            s=dot_size,
        )
        axs[i].set_xlabel(feature_labels[f1])
        axs[i].set_ylabel(feature_labels[f2])
        axs[i].set_zlabel("Price")
        axs[i].set_title(f"Lambda {lambda_:.1f}")

    # Add overall title and save the figure
    plt.suptitle(
        f"Price Prediction for {feature_labels[f1]} vs {feature_labels[f2]} for Different Lambda Values on best degree {best_degree}"
    )
    plt.tight_layout()
    plt.savefig(
        f"results/ex07/best_degree_prediction_{feature_labels[f1]}_{feature_labels[f2]}.png"
    )
    plt.close(fig)
