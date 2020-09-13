# House Price Regressions: A Comparative Study

### Introduction
This notebook provides a comparative study of three different regression techniques used in predicting the sale price of a house based on house features from Kaggle's Iowa Housing data set
(https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

### Overview
The challenge data set is intentionally messy. As such, the first step consisted of data cleaning and preprocessing: dealing with missing entries, creating dummy variables for
categorical types, and eliminating outliers. 

Next, Seaborn heatmaps were used to determine a set of features that were most strongly correlated with the sale price. These features were: 'OverallQual', 'GrLivArea', 'YearBuilt', '1stFlrSF', 'TotalBsmtSF', 'GarageArea', and 'GarageCars'. These values were then normalized using an instance of StandardScaler. 

The three regression techniques applied to the processed data were: ordinary least squares, random forest, and gradient boosting. These models were fitted using objects from the linear_model and ensemble packages from sklearn.

### Results
The table below summarizes the RMSE scores of each model, as applied to the training data set. 

Regression  | RMSE |
 :------------ | :----------- |
Ordinary Least Squares        |   0.156   | 
Random Forest  | 0.062  |
Gradient Boosting  | 0.071  |

As expected, the ordinary least squares regression provides the least predictive power, whereas the more sophisticated approaches of random forest and gradient regressions perform significantly better; obtaining root mean squared errors of less than 0.1 on the training data.

### How to Run the Code
To run this code, simply run the ipybn notebook file in Jupyter. Ensure that you have the following libraries installed: 	

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
```

### Acknowledgments

* [Pedro Marcelino's Kaggle kernel](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
* [Gerald Muriuki's repo](https://github.com/itsmuriuki/Predicting-House-prices)


### Contact information 

For any communication relating to this project, please email us at contact@thetaprime.io.

![alt text](thetaprime_shape.png)
