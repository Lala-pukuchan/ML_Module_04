import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex07.data_spliter import data_spliter
from ex07.polynomial_model import add_polynomial_features
from ex08.my_logistic_regression import MyLogisticRegression as MyRidge
from ex09.other_metrics import f1_score_
import pickle


def benchmark():
    # Store performance of models
    models = {}

    # set degree and lambda
    degree = 3
    lambda_values = np.arange(0, 1.1, 0.2)

    # load data
    solar_system_census = pd.read_csv("ex09/resources/solar_system_census.csv")
    x = np.array(solar_system_census[["weight", "height", "bone_density"]])
    solar_system_census_planets = pd.read_csv(
        "ex09/resources/solar_system_census_planets.csv"
    )
    y = np.array(solar_system_census_planets[["Origin"]])

    # split data
    np.random.seed(42)
    x_train_val, x_test, y_train_val, y_test = data_spliter(x, y, 0.85)
    x_train, x_cv, y_train, y_cv = data_spliter(x_train_val, y_train_val, 0.85)

    # Normalization
    min = x_train.min(axis=0)
    max = x_train.max(axis=0)
    base = max - min
    x_train_normalized = (x_train - min) / base
    x_cv_normalized = (x_cv - min) / base
    x_test_normalized = (x_test - min) / base

    # add polynomial feature with degree 3
    x_train_poly = add_polynomial_features(x_train_normalized, degree)
    x_cv_poly = add_polynomial_features(x_cv_normalized, degree)
    x_test_poly = add_polynomial_features(x_test_normalized, degree)

    # loop with each lambda
    for l in lambda_values:
        # initiate model
        model = MyRidge(
            theta=np.zeros((x_train_poly.shape[1] + 1, 1)),
            alpha=1e-2,
            max_iter=100_000,
            lambda_=l,
        )

        # storage for the probabilities of different zipcode
        predicted_probabilities_dict = {}

        # loop with each zipcode
        for zipcode in range(0, 5):
            # create data-set for one vs others
            y_train_binary = np.where(y_train == zipcode, 1, 0).reshape(-1, 1)

            # train model
            model.fit_(x_train_poly, y_train_binary)

            # store probabilities
            predicted_probabilities_dict[zipcode] = model.predict_(x_cv_poly)

        # Stack the probabilities horizontally
        all_probabilities = np.column_stack(
            [predicted_probabilities_dict[zipcode] for zipcode in range(5)]
        )

        # Store the class which has the highest probabilities
        y_cv_hat = np.argmax(all_probabilities, axis=1)

        # change y_cv from float to int
        y_cv = y_cv.astype(int).flatten()

        # get f1 score
        f1 = f1_score_(y_cv, y_cv_hat)

        # print f1 score
        print(
            f"lambda: {l:.2f}, f1 score: {f1:.2f}, Accuracy: {np.sum(y_cv == y_cv_hat) / len(y_cv):.2f}"
        )

        # Store model and performance
        models[l] = {
            "model": model,
            "f1": f1,
            "x_train_poly": x_train_poly,
            "y_train": y_train,
            "x_test_poly": x_test_poly,
            "y_test": y_test,
        }

    # Save the models to a file
    with open("ex09/models.pkl", "wb") as file:
        pickle.dump(models, file)


def main():
    benchmark()


if __name__ == "__main__":
    main()
