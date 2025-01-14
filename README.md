# readme

# Combined Cycle Power Plant Predictor
* Predict the **electrical energy output** of a combined cycle power plant  based on average ambient environmental readings 

## Data Source
[Combined cycle power plant readings](https://docs.google.com/spreadsheets/d/1B93JI2ROl8i4NBF-JIMrblLbi4YoEMqwwM5C_K8fAN8/edit?gid=1543908379#gid=1543908379)

### Rows (Observations)
* 9568 hourly average ambient environmental readings from sensors at the power plant which we will use in our model

### Columns (Features and Target)
* **Temperature (AT)** in the range 1.81°C to 37.11°C,
* **Exhaust Vacuum (V)** in the range 25.36-81.56 cm Hg
* **Ambient Pressure (AP)** in the range 992.89-1033.30 milibar,
* **Relative Humidity (RH)** in the range 25.56% to 100.16%
* **Net hourly electrical energy output(PE)**  (Target we are trying to predict)

## Outcome
* Predict energy output produced in any given time window

## Output
### Regression error metrics
* Maximum accuracy of energy output produced based on environment variables
* Use **Mean Absolute Percentage Error (MAPE)** to choose model that minimizes the total error

## Approach
* Determine what type of machine learning approach is needed
    * The data is *Continuous time series* (hourly)
    * Since we have historical data with target data, we can utilize a *supervised learning approach*
    * Since we are predicting numerical target values, we will take a *regression approach*

## Model Proposals
* ### Features
    * Since all our features are environmental readings that have been identified as key to determine PE, we will include all of them in our modeling datasets
    
    **Average ambient variables**
    * Temperature (AT)** in the range 1.81°C to 37.11°C,
    * Exhaust Vacuum (V)** in the range 25.36-81.56 cm Hg
    * Ambient Pressure (AP)** in the range 992.89-1033.30 milibar,
    * Relative Humidity (RH)** in the range 25.56% to 100.16%

* ### Target
    * **Net hourly electrical energy output(PE)**

* ### Algorithms
    * Since we have a known form to model and a small data set, use *parametric algorithms*
        * Since we have multiple features, we validate with *multiple linear regression*
        * Since we are not performing feature selection, we validate with *Ridge regression* as an alternative model to introduce regularization
        * As an alternative algorithm for comparison, validate with *Elastic-Net regression* 

* ### Hyperparameter
    * For multiple linear regression, there are no hyperparameters
    * For Ridge and Elastic-Net, lambda is a hyperparameter that controls the strength of the penalty

* ### Loss (Cost) function
    * Mean Squared Error (MSE)
    * Sum of Squared Errors (SSE)
    * Mean Absolute Percentage Error (MAPE)
    * Potentially *Regularize* our *cost function* to add more balance

## Train & Validate
* Use 80% of our observations for training and validation (7654 observations)
* Use K-Folds cross validation for model comparison and generalizing on new data

## Test
* We will use the remaining 20% of our observations for our test set (1914 observations)

## Algorithm Model Validation Results
#### **LinearRegression Cross-validated Results**
    * Fold MAPE: [0.81830992 0.80267334 0.78563095 0.79948492 0.79865026]
    * Mean MAPE: 0.8009498795266584
    * Accuracy: 99.19905012047334

####  **Ridge Cross-validated Results**
    * Fold MAPE: [0.81831516 0.8026979  0.78566237 0.79949961 0.798692  ]
    * Mean MAPE: 0.8009734067711698
    * Accuracy: 99.19902659322884

#### **ElasticNet Cross-validated Results**
    * Fold MAPE: [0.81830916 0.80268077 0.78563706 0.79948744 0.79865777]
    * Mean MAPE: 0.8009544401846747
    * Accuracy: 99.19904555981532

### **Highest Performing Model is LinearRegression**
    * MAPE: 0.8009498795266584
    * Accuracy: 99.19905012047334

## Test Predictions
    # Coefficients:  [-1.98589969 -0.23209358  0.06219991 -0.15811779]  
    # Intercept:  454.56911458941454  
    # MAPE:  0.7932984848767789  
    # Accuracy:  99.20670151512321 % 

## Final Model
* ### Features
    * Temperature (AT)** in the range 1.81°C to 37.11°C,
    * Exhaust Vacuum (V)** in the range 25.36-81.56 cm Hg
    * Ambient Pressure (AP)** in the range 992.89-1033.30 milibar,
    * Relative Humidity (RH)** in the range 25.56% to 100.16%

* ### Target
    * Net hourly electrical energy output(PE)

* ### Algorithm
    * Multiple Linear Regression

* ### Hyperparameter
    * None

* ### Loss function
    * Mean Absolute Percentage Error (MAPE) 

## Tools Used
* Google Sheets
* VSCode
* Python
* Python Libraries
    * Scikit
    * Pandas
    * Numpy