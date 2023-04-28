# OAE - Warning (trainee project)

## Introduction
Welcome to the OAE-Warning VAR trainee project! This project is a part of the larger OAE-Warning project, which aims to predict the yield and price of agricultural products in Thailand. In this particular project, we focus on predicting the price of rice using the Cobb Douglas and VAR models.

## What is the project about?
The primary goal of this project is to use the Cobb Douglas model to predict the yield of rice in Thailand. Once we have this prediction, we can use it as an independent variable in the VAR model to predict the price of rice. The VAR model is a statistical model that can predict multiple time series variables simultaneously.

## How does it work?
The Cobb Douglas model is a production function that calculates the output of a country's economy based on its inputs. In our case, we use it to predict the yield of rice in Thailand. The VAR model, on the other hand, is a statistical model that takes multiple time series variables as input and predicts their future values. By using the predicted yield of rice as one of the inputs, we can predict the future price of rice in Thailand.

## Models

Cobb Douglas and VAR models availble on ```sklean```

## Data

We retrive labour force , yield ,area of rice , price of oil and gas data from [bank of thailand](https://www.bot.or.th/) and price of product from [Office of Agricultural Economics](https://www.oaw.or.th).
