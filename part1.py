import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = "https://raw.githubusercontent.com/nifemiolayiwola/CS4375_Assignment1/refs/heads/main/concrete.csv"

df = pd.read_csv(DATA_URL)


print("First five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# Remove null/NA values
df = df.dropna()

# Remove redundant/duplicate rows
df = df.drop_duplicates()

print("\nRows after preprocessing:", len(df))


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


#  Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()



# Linear Regression using Gradient Descent
class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, iterations=1000):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = 0

        self.mse_history = []


    def predict(self, X):

        return np.dot(X, self.weights) + self.bias


    def mse(self, y_true, y_pred):

        return np.mean((y_true - y_pred) ** 2)


    def r2(self, y_true, y_pred):

        ss_res = np.sum((y_true - y_pred) ** 2)

        ss_total = np.sum(
            (y_true - np.mean(y_true)) ** 2
        )

        return 1 - (ss_res / ss_total)


    def explained_variance(self, y_true, y_pred):

        residuals = y_true - y_pred

        return 1 - (
            np.var(residuals) /
            np.var(y_true)
        )


    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        self.mse_history = []

        for i in range(self.iterations):

            # Predictions
            y_pred = self.predict(X)

            # Prediction error
            error = y_pred - y

            # Calculate gradients
            dw = (
                (2 / n_samples) *
                np.dot(X.T, error)
            )

            db = (
                (2 / n_samples) *
                np.sum(error)
            )

            # Update parameters
            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

            # Record MSE
            current_mse = self.mse(
                y,
                self.predict(X)
            )

            self.mse_history.append(
                current_mse
            )


# Parameter tuning
learning_rates = [
    0.0001,
    0.001,
    0.005,
    0.01,
    0.05
]

iterations_list = [
    500,
    1000,
    3000,
    5000
]

results = []


for lr in learning_rates:

    for iterations in iterations_list:

        model = LinearRegressionGD(
            learning_rate=lr,
            iterations=iterations
        )

        model.fit(
            X_train,
            y_train
        )

        train_predictions = model.predict(
            X_train
        )

        test_predictions = model.predict(
            X_test
        )

        train_mse = model.mse(
            y_train,
            train_predictions
        )

        test_mse = model.mse(
            y_test,
            test_predictions
        )

        results.append({
            "learning_rate": lr,
            "iterations": iterations,
            "training_mse": train_mse,
            "test_mse": test_mse
        })

        print(
            "Learning Rate:",
            lr,
            "Iterations:",
            iterations,
            "Test MSE:",
            test_mse
        )


# Save trial log
results_df = pd.DataFrame(results)

results_df.to_csv(
    "gradient_descent_trials.csv",
    index=False
)

print("\nAll parameter trials:")
print(results_df)

# Find best parameters
best_result = results_df.loc[
    results_df["test_mse"].idxmin()
]

print("\nBest parameters:")
print(best_result)

best_lr = best_result[
    "learning_rate"
]

best_iterations = int(
    best_result["iterations"]
)


#  Train best model
best_model = LinearRegressionGD(
    learning_rate=best_lr,
    iterations=best_iterations
)

best_model.fit(
    X_train,
    y_train
)


#  Evaluate best model
test_predictions = best_model.predict(
    X_test
)

final_test_mse = best_model.mse(
    y_test,
    test_predictions
)

final_r2 = best_model.r2(
    y_test,
    test_predictions
)

final_explained_variance = (
    best_model.explained_variance(
        y_test,
        test_predictions
    )
)

print("\nFINAL MODEL RESULTS")
print("----------------------------")

print(
    "Best Learning Rate:",
    best_lr
)

print(
    "Best Iterations:",
    best_iterations
)

print(
    "Final Test MSE:",
    final_test_mse
)

print(
    "Final Test R2:",
    final_r2
)

print(
    "Final Explained Variance:",
    final_explained_variance
)

print("\nWeights:")
print(best_model.weights)

print("\nBias:")
print(best_model.bias)


#  MSE vs Iterations Plot
plt.plot(
    best_model.mse_history
)

plt.xlabel("Iteration")
plt.ylabel("MSE")

plt.title(
    "MSE vs Number of Iterations"
)

plt.tight_layout()

plt.savefig(
    "mse_vs_iterations.png"
)

plt.show()

# Actual vs Predicted Plot
plt.figure()

plt.scatter(
    y_test,
    test_predictions
)

plt.xlabel("Actual Concrete Strength")
plt.ylabel("Predicted Concrete Strength")
plt.title("Actual vs Predicted Concrete Strength")

plt.tight_layout()

plt.savefig(
    "actual_vs_predicted.png"
)

plt.show()

# Age vs Concrete Strength Plot
age_column = df.columns[7]

plt.figure()

plt.scatter(
    df[age_column],
    df.iloc[:, -1]
)

plt.xlabel("Age (days)")
plt.ylabel("Concrete Compressive Strength (MPa)")
plt.title("Age vs Concrete Compressive Strength")

plt.tight_layout()

plt.savefig(
    "age_vs_strength.png"
)

plt.show()