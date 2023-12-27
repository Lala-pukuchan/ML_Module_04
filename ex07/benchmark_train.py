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

# Split data with train, cross validation and test sets
x_train_val, x_test, y_train_val, y_test = data_spliter(x, y, 0.8)
x_train, x_cv, y_train, y_cv = data_spliter(x_train_val, y_train_val, 0.75)

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
    x_cv_poly = add_polynomial_features(x_cv, degree)

    # set parameters
    theta = np.zeros((x_train_poly.shape[1] + 1, 1))
    alpha = 1e-2
    max_iter = 100000

    # change lambda in range 0 to 1 with step 0.2
    for lambda_ in np.arange(0, 1.1, 0.2):

        # Initialize the model
        model = MyR(theta, alpha, max_iter)

        # Fit the model
        model.fit_(x_train_poly, y_train)

        # Evaluate the model
        cross_valid_mse = model.mse_(y_cv, model.predict_(x_cv_poly))

        # debug
        print("--- degree", degree, "---")
        print("- lambda_", lambda_, "-")
        print("cross_valid_mse :", cross_valid_mse)

        # Store model and performance
        models_performance[degree][lambda_] = {
            "model": model,
            "cross_valid_mse": cross_valid_mse,
            "x_test_poly": x_test_poly,
            "y_test": y_test,
        }


# Save the models to a file
with open("ex07/models.pkl", "wb") as file:
    pickle.dump(models_performance, file)


# Plotting MSE for each model
#degrees = list(models_performance.keys())
#test_mses = [models_performance[d][l]["test_mse"] for d in degrees for l in lambda_]
#train_mses = [models_performance[d][l]["train_mse"] for d in degrees for l in lambda_]
#plt.figure()
#plt.plot(degrees, test_mses, marker="o", linestyle="-", color="b", label="Test MSE")
#plt.xlabel("Polynomial Degree")
#plt.ylabel("MSE")
#plt.title("Model Evaluation - Test MSE vs Polynomial Degree")
#plt.legend()
#plt.savefig("ex10/result_ex10_figure-1.png")
#plt.close()

for degree in models_performance:
    lambdas = models_performance[degree].keys()
    mses = [models_performance[degree][l]['cross_valid_mse'] for l in lambdas]

    plt.plot(lambdas, mses, label=f'Degree {degree}')

plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.title('Model Evaluation Curve')
plt.legend()
plt.savefig("ex07/figure-1.png")
plt.show()
