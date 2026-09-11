import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    explained_variance_score
)

# Load dataset from public GitHub location
DATA_URL = "https://raw.githubusercontent.com/nifemiolayiwola/CS4375_Assignment1/refs/heads/main/concrete.csv"

df = pd.read_csv(DATA_URL)


# Inspect dataset

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


# 6. Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parameter tuning
learning_rates = [
    0.0001,
    0.001,
    0.005,
    0.01
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

        model = SGDRegressor(
            loss="squared_error",
            penalty=None,
            learning_rate="constant",
            eta0=lr,
            max_iter=iterations,
            tol=None,
            random_state=42
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

        train_mse = mean_squared_error(
            y_train,
            train_predictions
        )

        test_mse = mean_squared_error(
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


#  Save trial log
results_df = pd.DataFrame(results)

results_df.to_csv(
    "sgd_trials.csv",
    index=False
)

print("\nAll SGDRegressor trials:")
print(results_df)


#  Find best parameters
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
best_model = SGDRegressor(
    loss="squared_error",
    penalty=None,
    learning_rate="constant",
    eta0=best_lr,
    max_iter=best_iterations,
    tol=None,
    random_state=42
)

best_model.fit(
    X_train,
    y_train
)


train_predictions = best_model.predict(
    X_train
)

test_predictions = best_model.predict(
    X_test
)


#  Evaluate best model
training_mse = mean_squared_error(
    y_train,
    train_predictions
)

test_mse = mean_squared_error(
    y_test,
    test_predictions
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

test_explained_variance = explained_variance_score(
    y_test,
    test_predictions
)


print("\nFINAL PART 2 RESULTS")
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
    "Training MSE:",
    training_mse
)

print(
    "Test MSE:",
    test_mse
)

print(
    "Test R^2:",
    test_r2
)

print(
    "Explained Variance:",
    test_explained_variance
)

print("\nModel Coefficients:")
print(best_model.coef_)

print("\nModel Intercept:")
print(best_model.intercept_)


# Actual vs Predicted Plot
plt.figure()

plt.scatter(
    y_test,
    test_predictions
)

plt.xlabel("Actual Concrete Strength")
plt.ylabel("Predicted Concrete Strength")
plt.title("SGDRegressor: Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "part2_actual_vs_predicted.png"
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
    "part2_age_vs_strength.png"
)

plt.show()


#  Model Coefficients Plot
plt.figure()

plt.bar(
    range(len(best_model.coef_)),
    best_model.coef_
)

plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("SGDRegressor Model Coefficients")

plt.tight_layout()

plt.savefig(
    "part2_coefficients.png"
)

plt.show()