import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ex00.polynomial_model_extended import add_polynomial_features
from ex06.ridge import MyRidge as MyRidge
from ex07.data_spliter import data_spliter


# # Store performance of models
# models_performance = {}

# # Train and evaluate models with polynomial features up to degree 4
# for degree in range(1, 5):
#     models_performance[degree] = {}

#     # lambda_values = np.arange(0, 1.1, 0.2)

#     # for l in lambda_values:

#     # Store model and performance
#     # models_performance[degree][l] = {
#     models_performance[degree] = {
#         "model": model,
#         "train_mse": train_mse,
#         "cv_mse": cv_mse,
#         "x_test_poly": x_test_poly,
#         "y_test": y_test,
#     }


# # Save the models to a file
# with open("ex07/models.pkl", "wb") as file:
#     pickle.dump(models_performance, file)


# # Plotting MSE for each model
# for degree in models_performance:
#     lambdas = sorted(models_performance[degree].keys())
#     # test_mses = [models_performance[degree][l]["cv_mse"] for l in lambdas]
#     test_mses = [models_performance[degree]["cv_mse"]]

#     # Plotting the test MSE for each lambda
#     # plt.plot(lambdas, test_mses, label=f"Degree {degree}", marker="o")
#     plt.plot(degree, test_mses, "bo-")

# plt.xlabel("Lambda")
# plt.ylabel("CV MSE")
# plt.title("CV MSE vs Lambda for Different Polynomial Degrees")
# plt.legend()
# plt.savefig("ex07/figure-1.png")


def benchmark_train():
    # load data
    data = pd.read_csv("ex07/space_avocado.csv")
    x = np.array(data[["weight", "prod_distance", "time_delivery"]])
    y = np.array(data["target"]).reshape(-1, 1)

    # split data
    x_train_val, x_test, y_train_val, y_test = data_spliter(x, y, 0.8)
    x_train, x_cv, y_train, y_cv = data_spliter(x_train_val, y_train_val, 0.6)

    # data storage
    mse_values = []

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
    for degree in degrees:
        # add polynomial feature
        x_train_poly = add_polynomial_features(x_train_scaled, degree)
        x_test_poly = add_polynomial_features(x_test_scaled, degree)
        x_cv_poly = add_polynomial_features(x_cv_scaled, degree)

        # initiate ridge regression class
        thetas = np.random.rand(x_train_poly.shape[1] + 1, 1)
        lr = MyRidge(thetas, alpha=0.1, max_iter=1000)

        # train model
        lr.fit_(x_train_poly, y_train)

        # cross validate with normalized mse
        mse = lr.mse_(y_cv, lr.predict_(x_cv_poly))
        mse_values.append(mse / variance)

    plot_evaluation_curve(degrees, mse_values)


def plot_evaluation_curve(degrees, mse_values):
    plt.plot(degrees, mse_values, "bo-")
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Mean Squared Error")
    plt.title("Evaluation Curve")
    plt.savefig("ex07/evaluation_curve.png")


#def plot_predictions(model, x_test_poly, y_test):
#    print(x_test_poly.shape, y_test.shape)
#    y_pred = model.predict_(x_test_poly)

#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection="3d")
#    ax.scatter(x_test_poly[:, 1], x_test_poly[:, 2], y_test, c="b", label="True Price")
#    ax.scatter(
#        x_test_poly[:, 1], x_test_poly[:, 2], y_pred, c="r", label="Predicted Price"
#    )
#    ax.set_xlabel("Weight")
#    ax.set_ylabel("Production Distance")
#    ax.set_zlabel("Price")
#    ax.legend()
#    plt.title("True Price vs Predicted Price")
#    plt.show()

#    print(
#        x_test_poly.shape,
#    )
#    print(
#        x_test_poly,
#    )


if __name__ == "__main__":
    benchmark_train()
