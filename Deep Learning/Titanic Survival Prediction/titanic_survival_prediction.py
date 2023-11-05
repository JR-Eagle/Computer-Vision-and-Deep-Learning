import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def load_and_process_data():
    # Load dataset
    df = pd.read_csv("../DS3_titanic.csv")
    # Select necessary features and target variable
    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
    # Convert Sex column to numeric (1 if male, 0 otherwise)
    is_male = df["Sex"].apply(lambda x: 1 if x == 'male' else 0)
    df["IsMale"] = is_male
    # Impute missing age values with mean age
    ave_age = df["Age"].mean()
    df["Age"] = df["Age"].fillna(ave_age)
    return df

def train_model(df):
    # Define explanatory variables and target variable
    X = df[["Pclass", "Age", "Fare", "IsMale"]]
    y = df["Survived"]
    # Split data into training and testing sets (20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Create an instance of Decision Tree classifier (fixing random seed for reproducibility)
    model_full = DecisionTreeClassifier(random_state=0)
    # Set grid search parameters
    param_grid = {
        "max_depth": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10]
    }
    # Perform grid search to find the optimal model (using 4-fold CV)
    grid_search_dt = GridSearchCV(model_full, param_grid, cv=4)
    grid_search_dt.fit(X_train, y_train)
    return grid_search_dt, X_test, y_test

def evaluate_model(grid_search_dt, X_test, y_test):
    # Make predictions on the test set
    pred_y_test = grid_search_dt.predict(X_test)
    # Compare predicted labels to actual labels and compute accuracy
    prediction_result = (y_test == pred_y_test)
    accuracy = np.mean(prediction_result)
    # Print accuracy
    print("Accuracy of Survived Prediction: {:.1f}%".format(accuracy*100))

if __name__ == "__main__":
    df = load_and_process_data()
    grid_search_dt, X_test, y_test = train_model(df)
    evaluate_model(grid_search_dt, X_test, y_test)
