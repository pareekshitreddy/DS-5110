# Predictive Analysis of Hotel Booking Data[^1^][1]

This project aims to apply data analytics techniques to a hotel booking dataset and provide insights and predictions for both customers and the hotel industry. The dataset contains booking information for a city hotel and a resort hotel over the period 2015-2017, with over 100k records and 32 features.

## Project Goals

The project goals are:

- To analyze the best time of year to book a hotel room and the optimal length of stay to get the best daily rate.
- To classify customers based on who is more likely to cancel their booking.
- To predict the number of future guests for both hotel types.
- To predict the likelihood of a disproportionately high number of special requests.

## Data Preprocessing

The data preprocessing steps include:

- Dropping columns with a high percentage of null values ('company' and 'agent')
- Dropping rows with null values in the 'country' column
- Imputing null values in the 'children' column with zeros
- Converting columns to the correct data types
- Creating new columns from existing columns (e.g., total number of guests, total length of stay, arrival date)
- Label encoding categorical columns

## Exploratory Data Analysis

The exploratory data analysis aims to explore the trends and relationships among the variables. Some of the questions answered are:

- What is the average number of guests per month, and which season is the most preferred?
- What time of the year has the lowest booking rates?
- Does higher lead time result in cancellations?
- Do all guests who made changes in booking really check-in?
- Which customers are most likely to cancel their bookings?

The analysis also includes feature selection methods such as the correlation matrix and feature importance plot.

## Modelling

The modelling part involves three different prediction tasks:
1. **Predicting whether a booking made by a customer would be cancelled or not.** This is a binary classification problem, and the models used are Logistic Regression, Support Vector Machine, XGBoost, and LightGBM.
2. **Predicting the likelihood of a disproportionately high number of special requests.**
3. **Predicting the number of future guests for the hotel.**This is a time series forecasting problem, and the model used is ARIMA.

The models are evaluated using different metrics such as accuracy, precision, recall, F1-score, ROC-AUC, mean squared error, and mean absolute error.

## Results

The results show the performance of each model on the prediction tasks and the comparison among them. The best models are:

- **XGBoost** for predicting whether a booking would be cancelled or not, with an accuracy of 0.86 and an ROC-AUC of 0.93.
- **XGBoost** for predicting the likelihood of a disproportionately high number of special requests, with a mean squared error of 0.17 and a mean absolute error of 0.28.
- **ARIMA** for predicting the number of future guests for the hotel, with a mean squared error of 0.01 and a mean absolute error of 0.07.

## Discussion

The discussion section summarizes the impact of the results on both customers and the hotel industry. Some of the impacts are:

- Customers can use the analysis to decide the best time of year to book a hotel room and the optimal length of stay to get the best daily rate.
- The hotel industry can use the analysis to understand the factors affecting customer decisions and cancellations, and improve their marketing strategy and customer loyalty.
- The hotel industry can also use the predictions to better prepare for the demand and manage the required resources to ensure customer satisfaction.

## Future Work

The future work section outlines some possible improvements and extensions for the project. Some of the future work includes:

- Implementing more algorithms like Catboost and neural networks.
- Using GridSearchCV and Bayesian Optimization for tuning the hyperparameters of various models.
- Collecting more data in the future to improve the model accuracy and generalization.
- Ensembling complex models such as LightGBM and XGBoost.


