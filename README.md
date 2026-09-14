# CS4375_Assignment1
# Linear Regression using Gradient Descent

The project uses the Concrete Compressive Strength dataset from the UCI Machnine Learning Repository

The dataset is hosted on Github publicly: https://github.com/nifemiolayiwola/CS4375_Assignment1

How to Compile and Run the code
1. Download the submitted files and put them in the same folder

2. Open the folder in VS Code or a terminal

3. Install the required libraries
pip install numpy pandas matplotlib scikit-learn

4. Run part 1 on your terminal
python3 part1.py

(Part 1 implements linear regression using my gradient descent algorithm)

5. Run part 2
python3 part2.py

(Part 2 uses Scikit-learn's SGDRegressor for linear regression using gradient descent)

Libraries used:
1. pandas: to load and preprocess the dataset
2. matplotlib: to create plots
3. scikit-learn: 
 - train_test_split - splitting training and test data
 - StandardScaler - standardizing features
 - SGDRegressor - linear regression using gradient descent
 - mean_squared_error - calculating MSE
 - r2_score - calculating R^2
 - explained_variance_score - calculating explained variance


 Files:
 part1.py
 part2.py
 gradient_descent_trials.csv
 sgd_trials.csv
 README.md