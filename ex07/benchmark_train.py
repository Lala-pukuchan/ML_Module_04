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


def evaluate_model(model, x_test_poly, y_test):
    # Make predictions on the test data
    y_pred = model.predict_(x_test_poly)

    # Calculate the mean squared error
    mse = np.mean((y_test - y_pred) ** 2)
    return mse


def benchmark_train(degrees):
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

    # Find the best degree with the minimum mean squared error
    print(degrees, mse_values)
    best_degree = degrees[np.argmin(mse_values)]
    print("Best Degree:", best_degree)

    # Train the best model based on the best degree using the combined training and cross-validation sets
    X_train_cv_poly = add_polynomial_features(
        np.vstack((x_train_scaled, x_cv_scaled)), best_degree
    )
    # Concatenate y_train and y_cv
    y_train_cv = np.concatenate((y_train, y_cv))

    x_test_poly = add_polynomial_features(x_test_scaled, best_degree)
    num_features = X_train_cv_poly.shape[1] + 1
    thetas = np.random.rand(num_features, 1)
    best_model = MyRidge(thetas, alpha=0.1, max_iter=1000)
    # Use y_train_cv instead of np.vstack((y_train, y_cv))
    best_model.fit_(X_train_cv_poly, y_train_cv)

    # Save the parameters of all the models into a file (models.csv)
    models = {"degrees": degrees, "mse_values": mse_values}
    models_df = pd.DataFrame(models)
    models_df.to_csv("ex07/models.csv", index=False)
    print("Data saved to models.csv file.")

    return mse_values, best_model, x_test_poly, y_test


def plot_evaluation_curve(degrees, mse_values):
    plt.plot(degrees, mse_values, "bo-")
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Mean Squared Error")
    plt.title("Evaluation Curve")
    plt.show()


def plot_predictions(model, x_test_poly, y_test):
    print(x_test_poly.shape, y_test.shape)
    y_pred = model.predict_(x_test_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_test_poly[:, 1], x_test_poly[:, 2], y_test, c="b", label="True Price")
    ax.scatter(
        x_test_poly[:, 1], x_test_poly[:, 2], y_pred, c="r", label="Predicted Price"
    )
    ax.set_xlabel("Weight")
    ax.set_ylabel("Production Distance")
    ax.set_zlabel("Price")
    ax.legend()
    plt.title("True Price vs Predicted Price")
    plt.show()

    print(
        x_test_poly.shape,
    )
    print(
        x_test_poly,
    )


degrees = [1, 2, 3, 4]

mse_values, best_model, x_test_poly, y_test = benchmark_train(degrees)

# Plot the evaluation curve
plot_evaluation_curve(degrees, mse_values)


# load the saved

# Plot the true price and predicted price using the best model
plot_predictions(best_model, x_test_poly, y_test)
