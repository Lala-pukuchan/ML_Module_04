import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ex07.data_spliter import data_spliter
from ex07.polynomial_model import add_polynomial_features
from ex08.my_logistic_regression import MyLogisticRegression as MyRidge
from ex09.other_metrics import f1_score_
import pickle


def benchmark():

    # set lambda
    lambda_values = np.arange(0, 1.1, 0.2)

    # Load the data and models from the pickle file
    with open("ex09/models.pkl", "rb") as file:
        loaded_models = pickle.load(file)

    # storage for f1 score
    f1_scores = []

    # initial f1 score
    min_f1 = 0

    # loop with each lambda
    for l in lambda_values:
        # take model
        model_info = loaded_models[l]

        # initiate model
        model = model_info["model"]

        # storage for the probabilities of different zipcode
        predicted_probabilities_dict = {}

        # take data used for train and test
        x_train_poly = model_info["x_train_poly"]
        y_train = model_info["y_train"]
        x_test_poly = model_info["x_test_poly"]
        y_test = model_info["y_test"]

        # loop with each zipcode
        for zipcode in range(0, 5):
            # create data-set for one vs others
            y_train_binary = np.where(y_train == zipcode, 1, 0).reshape(-1, 1)

            # train model
            model.fit_(x_train_poly, y_train_binary)

            # store probabilities
            predicted_probabilities_dict[zipcode] = model.predict_(x_test_poly)

        # Stack the probabilities horizontally
        all_probabilities = np.column_stack(
            [predicted_probabilities_dict[zipcode] for zipcode in range(5)]
        )

        # Store the class which has the highest probabilities
        y_test_hat = np.argmax(all_probabilities, axis=1)

        # change y_test from float to int
        y_test = y_test.astype(int).flatten()

        # get f1 score
        f1 = f1_score_(y_test, y_test_hat)
        f1_scores.append(f1)

        # print f1 score
        print(
            f"lambda: {l:.2f}, f1 score: {f1:.2f}, Accuracy: {np.sum(y_test == y_test_hat) / len(y_test):.2f}"
        )

        # plot the results for best model
        if f1 > min_f1:
            plt.figure(figsize=(10, 6))
            correct = y_test == y_test_hat
            plt.scatter(x_test_poly[correct, 0], x_test_poly[correct, 1], c='green', label='Correct Prediction', alpha=0.5)
            incorrect = ~correct
            plt.scatter(x_test_poly[incorrect, 0], x_test_poly[incorrect, 1], c='red', label='Incorrect Prediction', alpha=0.5)
            plt.xlabel('weight')
            plt.ylabel('height')
            plt.title(f'Predictions of the Best Model (λ={l:.2f})')
            plt.legend()
            plt.grid(True)
            plt.savefig("./results/ex09/best_model_predictions.png")

    # plot bar graph for f1 score and each lambda
    plt.figure(figsize=(10, 6))
    plt.bar(lambda_values, f1_scores, width=0.1)
    plt.xlabel("Lambda")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Lambda")
    plt.xticks(lambda_values)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("./results/ex09/f1_score_bar.png")


def main():
    benchmark()


if __name__ == "__main__":
    main()
