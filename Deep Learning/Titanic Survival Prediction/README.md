# Titanic Survival Prediction

This project utilizes the Titanic dataset to predict passenger survival based on various features such as class, age, fare, and gender. The prediction model is built using a Decision Tree classifier, with hyperparameters tuned via grid search and 4-fold cross-validation.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

## Data Preparation

The `load_and_process_data` function reads the Titanic dataset from `DS3_titanic.csv`, selects necessary features and the target variable, converts the `Sex` column to numeric, and imputes missing age values with the mean age.

## Model Training

The `train_model` function defines the explanatory variables and target variable, splits the data into training and testing sets, creates an instance of a Decision Tree classifier, sets grid search parameters for hyperparameter tuning, and performs grid search to find the optimal model using 4-fold cross-validation.

## Model Evaluation

The `evaluate_model` function makes predictions on the test set using the trained model, compares predicted labels to actual labels, computes the accuracy of predictions, and prints the accuracy.

## Results
When run locally, the script achieved an accuracy of 82.1% on predicting survival on the Titanic.

## Usage

Ensure you have all the necessary dependencies installed, then run the script using:

```
python script_name.py
```
