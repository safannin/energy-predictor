"""Combined Cycle Power Plant Energy Predictor 

Machine learning tool that takes in ambient environment variables and predicts the electrical energy output
"""
from math import sqrt
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from pytest import console_main
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model, tree
from sklearn.preprocessing import PolynomialFeatures

SCORER = "neg_mean_squared_error"
SCORER_LABEL= "MSE"

def sum_of_squared_errors(y_true, y_pred):
    """Calculates the Sum of Squared Errors (SSE)."""
    return np.sum((y_true - y_pred)**2)

def load_data_from_file (filenamepath):
    """Load the CSV file"""
    return pd.read_csv(filenamepath)

def prep_training_test_sets (data):
    """Split our data into training and test sets.  Assumes last column is the target variable"""
    features = data.iloc[:, :-1]  # Features
    target = data.iloc[:, -1]   # Target variable
    return train_test_split(features, target, test_size=0.2, random_state=42)

def cross_validate(model, X_train, y_train, cv, scorer, poly=False):
    """Use cross-validation to validate and predict how the model will handle unknown data"""
    train_scores = abs(cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer))
    mean_train_scores = np.mean(train_scores)
    model_name = (f"Polynomial {type(model).__name__}") if poly is True else (f"{type(model).__name__}")
    print_results([[f"Folds {SCORER_LABEL}", train_scores], 
                   [f"Mean Folds {SCORER_LABEL}", mean_train_scores],
                   [f"Mean Folds Root {SCORER_LABEL}", sqrt(mean_train_scores)]], title=model_name)
    return [model, mean_train_scores, model_name]

def predict_with_metrics (model, x_test, y_test):
    """Predict the model and show key metrics"""
    y_pred = model.predict(x_test)
    coeff = model.coef_ if hasattr(model, "coef_") else "Not applicable"
    intercept = model.intercept_ if hasattr(model, "intercept_") else "Not applicable"
    mse = mean_squared_error(y_test,y_pred)
    #mae = mean_absolute_error(y_test,y_pred)
    #mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    #accuracy = (1-(mape/100))*100
    #r2 = r2_score(y_test, y_pred)
    #sse = sum_of_squared_errors(y_test, y_pred)
    #kv_coeff = ["Coefficients", coeff]
    values = [["Coefficients", coeff], ["Intercept", intercept], 
              [SCORER_LABEL, mse], [f"Root {SCORER_LABEL}",sqrt(mse)]]

    print_results(values, f"Predictions with Test Data using {type(model).__name__}")
    # print (f"\nCoefficients: {coeff}\r")
    # print (f"Intercept: {intercept}\r")
    # print (f"{SCORER_LABEL}: {mse}\r")
    # print (f"Root {SCORER_LABEL}: {sqrt(mse)}\r")
    #print ("Accuracy: ", accuracy, "% \r")
    #print ("Sum of squared errors: ", sse, " \r")

    # print ("Mean squared error: ", mse, " \r")
    # print ("Mean absolute error: ", mae, " \r")
    
    # print ("R2 score: ", r2, " \r")   

def determine_best_model (models):
    """Determine and return the optimal model from a list of proposed models by comparing the lowest MAPE value"""
    best = models[0]

    for mod in models:
        if mod[1] < best[1]:
            best = mod
    
    return best

def print_results(kv_results, title="Results"):
    """Print results 

    Args:
        title: Title line of results
        kv_results (list): key-value pair of results where key is label and value represents result value
    """
    print (f"\n---{title}---\r")
    for result in kv_results:
        print (f"{result[0]}: {result[1]}\r")

#Clear console
print("\033[H\033[J", end="")

#Load source data
_plant_data = load_data_from_file("../CCPP_data.csv")

# Split the data into training and testing sets
_X_train, _X_test, _y_train, _y_test = prep_training_test_sets(_plant_data)

#Use cross-validation with different models to see which model optimizes performance
#Set-up K-Fold cross validation and scoring approach
_kf = KFold(n_splits=5, shuffle=True, random_state=42)
print ("\n---Use cross-validation to validate and predict how the model will handle unknown data--")

#Linear Regression
_lin_reg_model = linear_model.LinearRegression()

#Cross validate model and report results
_lin_scores = cross_validate(_lin_reg_model, _X_train, _y_train, _kf, SCORER)
_model_scores = [_lin_scores]

#Ridge Regression
#Determine optimal lambda value
ridge_cv = linear_model.RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
ridge_cv.fit (_X_train, _y_train)
#Cross validate model and report results
_ridge_model = linear_model.Ridge(alpha=ridge_cv.alpha_)
_ridge_scores = cross_validate(_ridge_model, _X_train, _y_train, _kf, SCORER)
_model_scores.append(_ridge_scores)

#Elastic-Net Regression
#Determine optimal lambda value
elastic_cv = linear_model.ElasticNetCV(alphas=np.logspace(-4, 4, 100), cv=5)
elastic_cv.fit (_X_train, _y_train)
#Cross validate model and report results
_elastic_model = linear_model.ElasticNet(alpha=elastic_cv.alpha_)
_elastic_scores = cross_validate(_elastic_model, _X_train, _y_train, _kf, SCORER)
_model_scores.append(_elastic_scores)

#Polynomial Regression
_poly = PolynomialFeatures(2)
_x_poly = _poly.fit_transform(_X_train)
#Cross validate model and report results
_poly_lin_scores = cross_validate(_lin_reg_model, _x_poly, _y_train, _kf, SCORER, poly=True)
_model_scores.append(_poly_lin_scores)

#Decision Trees
_dtrees = tree.DecisionTreeRegressor(max_depth=8)
_dtrees_scores = cross_validate(_dtrees, _X_train, _y_train, _kf, SCORER)
_model_scores.append(_dtrees_scores)

#Random Forest
_forest = RandomForestRegressor(n_estimators=10)
_forest_scores = cross_validate(_forest,  _X_train, _y_train, _kf, SCORER)
_model_scores.append(_forest_scores)

#Top Performer
winner = determine_best_model(_model_scores)
print_results([[SCORER_LABEL,winner[1]], [f"Root {SCORER_LABEL}", sqrt(winner[1])]],
              title=f"Highest Performing Model is {winner[2]}")

#Run predictions and determine key metrics for selected model
print ("\n---Run predictions and determine key metrics for selected model--")
winner[0].fit(_X_train, _y_train)
predict_with_metrics(winner[0], _X_test, _y_test)

print("\n")