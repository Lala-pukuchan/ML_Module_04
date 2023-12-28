import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ex00.polynomial_model_extended import add_polynomial_features
from ex06.ridge import MyRidge as MyRidge
from ex07.data_spliter import data_spliter


def benchmark_train():
    # load data
    data = pd.read_csv("ex07/space_avocado.csv")
    x = np.array(data[["weight", "prod_distance", "time_delivery"]])
    y = np.array(data["target"]).reshape(-1, 1)

    # split data
    x_train_val, x_test, y_train_val, y_test = data_spliter(x, y, 0.8)
    x_train, x_cv, y_train, y_cv = data_spliter(x_train_val, y_train_val, 0.75)

    # Store performance of models
    models_performance = {}

    # Normalization
    min = x_train.min(axis=0)
    max = x_train.max(axis=0)
    base = max - min
    x_train_scaled = (x_train - min) / base
    x_cv_scaled = (x_cv - min) / base
    x_test_scaled = (x_test - min) / base

    # variance of y
    variance = np.var(y_train)

    # loop with each degree
    degrees = [1, 2, 3, 4]
    smallest_mse = float("inf")
    for degree in degrees:

        # storage for each degree
        models_performance[degree] = {}

        # add polynomial feature
        x_train_poly = add_polynomial_features(x_train_scaled, degree)
        x_test_poly = add_polynomial_features(x_test_scaled, degree)
        x_cv_poly = add_polynomial_features(x_cv_scaled, degree)

        # loop with each lambda
        lambda_values = np.arange(0, 1.1, 0.2)
        for l in lambda_values:
            # initiate ridge regression class
            thetas = np.random.rand(x_train_poly.shape[1] + 1, 1)
            model = MyRidge(thetas, alpha=0.1, max_iter=1000, lambda_=l)

            # train model
            model.fit_(x_train_poly, y_train)

            # cross validate with normalized mse
            mse = model.mse_(y_cv, model.predict_(x_cv_poly)) / variance

            # print
            print(f"degree: {degree}, lambda: {l}, mse: {mse}")
            if (mse < smallest_mse):
                smallest_mse = mse
                best_degree = degree
                best_lambda = l

            # Store model and performance
            models_performance[degree][l] = {
                "model": model,
                "mse": mse,
                "x_test_poly": x_test_poly,
                "y_test": y_test,
            }

    # Print the best model
    print(f"Best model: degree {best_degree}, lambda {best_lambda}, mse {smallest_mse}")
    
    # Save the models to a file
    with open("ex07/models.pkl", "wb") as file:
        pickle.dump(models_performance, file)

def plot_evaluation_curve():
    # Open the file and load the data
    with open("ex07/models.pkl", "rb") as file:
        models_performance = pickle.load(file)

    # Plotting MSE for each model
    for degree in models_performance:
        lambdas = sorted(models_performance[degree].keys())
        test_mses = [models_performance[degree][l]["mse"] for l in lambdas]

        # Plotting the test MSE for each lambda
        plt.plot(lambdas, test_mses, label=f"Degree {degree}", marker="o")

    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    plt.title("MSE vs Lambda for Different Polynomial Degrees")
    plt.legend()
    plt.savefig("ex07/evaluation_curve.png")


if __name__ == "__main__":
    # train model and save to file
    benchmark_train()
    # plot evaluation curve for all models
    plot_evaluation_curve()
