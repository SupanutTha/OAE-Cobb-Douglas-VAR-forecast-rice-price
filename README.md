# OAE - Warning

## Introduction
Welcome to the OAE-Warning VAR trainee project! This project is a part of the larger OAE-Warning project, which aims to predict the yield and price of agricultural products in Thailand. In this particular project, i focus on predicting the price of rice using the Cobb Douglas and VAR models.

## What is the project about?
The primary goal of this project is to use the Cobb Douglas model to predict the yield of rice in Thailand. Once i have this prediction, i can use it as an independent variable in the VAR model to predict the price of rice. The VAR model is a statistical model that can predict multiple time series variables simultaneously.

## How does it work?
The Cobb Douglas model is a production function that calculates the output of a country's economy based on its inputs. In our case, i use it to predict the yield of rice in Thailand. The VAR model, on the other hand, is a statistical model that takes multiple time series variables as input and predicts their future values. By using the predicted yield of rice as one of the inputs, i can predict the future price of rice in Thailand.

## Methodology
The project is divided into two parts:

Part 1: Cobb-Douglas production function
The Cobb-Douglas production function is used to model the relationship between the inputs of production and the output. The inputs of production are land, labor, and capital, and the output is the yield of rice. The Cobb-Douglas production function is given by:

Y = A * K^α * N^β * M^γ

where:

Y is the yield of rice
K is the total area planted
N is the total area harvested
M is the total labor input
A is the total factor productivity
α, β, and γ are the parameters to be estimated
To estimate the parameters of the production function, i use the Ordinary Least Squares (OLS) method. The log of both sides of the production function is taken to linearize it. The resulting equation is:

ln(Y) = ln(A) + α * ln(K) + β * ln(N) + γ * ln(M)

fit this equation with the OLS method using the statsmodels library in Python.

Part 2: Vector Autoregression (VAR) model
The VAR model is a statistical model used to analyze the relationship between multiple time series variables. In this project, i use the VAR model to analyze the relationship between rice yield, rice exports, rice export volume, oil prices, gas prices, and rice prices.

## Data

We retrive labour force , yield ,area of rice , price of oil and gas data from [bank of thailand](https://www.bot.or.th/) and price of product from [Office of Agricultural Economics](https://www.oaw.or.th).

## Result

Cobb Douglas that use to predict the yield of rice

The model had an R-squared value of 0.9999982484371286.


Mean Squared Error of 162950111690.96857. 


The Root MSE was 403670.79618293984.


the Mean Absolute Error was 315941.84128418355.


the prediction is

| Year | Actual Yield | Predicted Yield | Percent Change |
|------|--------------|----------------|----------------|
| 2557 | 19,440,378   | 18,582,180     | -4.41%         |
| 2558 | 14,879,651   | 15,257,730     | 2.54%          |
| 2559 | 17,512,015   | 18,045,520     | 3.05%          |
| 2560 | 19,003,323   | 18,879,100     | -0.65%         |
| 2561 | 19,503,457   | 19,303,620     | -1.02%         |
| 2562 | 19,123,461   | 18,984,530     | -0.73%         |
| 2563 | 20,166,081   | 20,416,940     | 1.24%          |
| 2564 | 20,513,907   | 20,557,810     | 0.21%          |


From this table, you can see that the percent change varies from year to year. In some years, such as 2557 and 2558, the predicted yield was lower than the actual yield, resulting in negative percent changes. In other years, such as 2559 and 2563, the predicted yield was higher than the actual yield, resulting in positive percent changes. Overall, the average percent change over all years is close to zero, indicating that the predicted yield is generally accurate but has some variability from year to year.

but in VAR model's result is not satisfied.
R-squared: -8.460544118500938
MSE: 3858949.2625019625
MAE: 1777.639910894819
The negative value of R-squared indicates that the model is not a good fit for the data.
The MSE and MAE values indicate the magnitude of the error in the predictions, with higher values indicating larger errors.

## Conclusion
This project aimed to predict the price of rice in Thailand using the Cobb Douglas and VAR models. The Cobb Douglas model was used to predict the yield of rice in Thailand, which was then used as an independent variable in the VAR model to predict the future price of rice.

The results of the Cobb Douglas model showed that the predicted yield of rice was generally accurate, but had some variability from year to year. The VAR model, however, did not produce satisfactory results. The negative value of R-squared indicated that the model was not a good fit for the data, and the MSE and MAE values indicated a high magnitude of error in the predictions.

p.s. i created VAR models that use month time frame data but the result is still not acceptable.
