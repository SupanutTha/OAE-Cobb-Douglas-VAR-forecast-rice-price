import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
lab = pd.read_csv('/Users/markio/oae/research/lab.csv')
area = pd.read_excel('/Users/markio/oae/research/ข้าว_พื่นที่ปลูก_ผลผลิต64-57.xlsx')
rain = pd.read_excel('/Users/markio/oae/research/rainfall.xlsx')
price = pd.read_excel('/Users/markio/oae/research/ราคาข้าว.xlsx')

# Data preparation
lab = lab.set_index('year')
area['Type'].fillna('', inplace=True)

area_filtered = area.loc[area['Type'].str.contains('ข้าวเจ้า')].drop(['City', 'Type'], axis=1)
area_filtered.columns = ['year', 'area_planted', 'area_harvested', 'yield']
area_filtered = area_filtered.apply(pd.to_numeric, errors='coerce')

# Aggregate data by year
total_area_planted = area_filtered.groupby('year')['area_planted'].sum()
total_area_harvested = area_filtered.groupby('year')['area_harvested'].sum()
total_yield = area_filtered.groupby('year')['yield'].sum()

# Test field
rain = rain.drop(columns= ['Sum of MinRain', 'Sum of MaxRain'])
rain = rain.set_index('year')
rain= rain.drop(index= 2565)

lab = lab.drop(index = [2554,2555,2556,2565])
lab['รวม'] = lab['รวม'].str.replace(',', '').astype(float)
# total_area_harvested = total_area_harvested.drop(index = [2559,2560])
# total_area_planted = total_area_planted.drop(index = [2559,2560])
# total_yield = total_yield.drop(index = [2559,2560])


# Cobb-Douglas production function
alpha = 0.5  # parameter to be estimated
Y = total_yield
X = pd.DataFrame({'K': total_area_planted, 'N': total_area_harvested ,'M': lab['รวม']})
X = sm.add_constant(X)  # add constant term for estimation

# Fit Cobb-Douglas function with OLS
model = sm.OLS(np.log(Y), np.log(X)).fit()
print(model.summary())
# reduce overfitting
# model = sm.OLS(np.log(Y), np.log(X)).fit_regularized(alpha=0.1, L1_wt=0.5)
# print(model.summary())


X_scaled = np.log(X)
Y_pred = np.exp(model.predict(X_scaled))
# print(Y)
# print(Y_pred)


# Calculate R-squared
r_squared = model.rsquared
print("R-squared:", r_squared)

# Calculate Mean Squared Error
mse = ((Y - Y_pred) ** 2).mean()
print("Mean Squared Error:", mse)

# Calculate Mean Absolute Error
mae = np.abs(Y - Y_pred).mean()
print("Mean Absolute Error:", mae)

# VAR

# import data

export = pd.read_excel('/Users/markio/oae/research/export_year.xlsx', index_col=0)
oil_gas = pd.read_excel('/Users/markio/oae/research/oil_year.xlsx', index_col=0)


# price = price['year'].astype(str)
Y_pred = pd.DataFrame(Y_pred).reset_index()
Y_pred = Y_pred.sort_values('year', ascending= False).reset_index().drop(columns='index')
riceEx = export.iloc[1]
riceEx_vol = export.iloc[2]
riceEx = pd.DataFrame(riceEx)
riceEx_vol = pd.DataFrame(riceEx_vol)
oil = oil_gas.iloc[1]
gas = oil_gas.iloc[10]
price = price.iloc[2:10]
riceEx.index.name = None

riceEx = riceEx.iloc[1:]
riceEx_vol = riceEx_vol.iloc[1:]
oil = oil.iloc[2:]
gas = gas.iloc[2:]
# test data
riceEx = pd.DataFrame(riceEx)
riceEx = riceEx.reset_index().drop(columns='index').apply(pd.to_numeric)
riceEx_vol = pd.DataFrame(riceEx_vol)
riceEx_vol = riceEx_vol.reset_index().drop(columns='index').apply(pd.to_numeric)
oil = pd.DataFrame(oil)
oil = oil.reset_index().drop(columns='index').apply(pd.to_numeric)
gas = pd.DataFrame(gas)
gas = gas.reset_index().drop(columns='index').apply(pd.to_numeric)


price = price.reset_index().drop(columns='index').drop(columns='year')
# print(Y_pred)
# print(riceEx)
# print(riceEx_vol)
# print(oil)
# print(gas)

# data = pd.concat([Y_pred, price, riceEx, riceEx_vol, oil, gas], axis=1)
# print(data)

# Combine the data
data = pd.concat([Y_pred, riceEx, riceEx_vol, oil, gas, price], axis=1)
data = data.set_index('year')

# Prepare the data
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# Train-test split
train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[:train_size, :], data
# Fit VAR model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(train_data)
results = model.fit()

# Make predictions
lag_order = results.k_ar
pred = results.forecast(train_data.values[-lag_order:], len(test_data))

print(lag_order)

# Evaluate the model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

r_squared = r2_score(test_data['price'], pred[:, -1])
print('R-squared:',r_squared)
mse = mean_squared_error(test_data['price'], pred[:, -1])
print('MSE:', mse)
mae = mean_absolute_error(test_data['price'], pred[:, -1])
print("Mean Absolute Error:", mae)

var_model = model.fit(lag_order)

prediction = var_model.forecast(data.values[-lag_order:], steps=1)
lag_order = results.k_ar
pred = results.forecast(train_data.values[-lag_order:], len(test_data))
pred = pd.DataFrame(pred, index=test_data.index, columns=test_data.columns)

print(data)
print(pred)
print(prediction)


model = VAR(data)
results = model.fit()

# Predict the next 2 years
n_periods = 2  # 2 years * 12 months
forecast = results.forecast(data.values[-results.k_ar:], n_periods)

# Convert the forecast to a DataFrame with the appropriate index
forecast_index = pd.date_range(start='2023-01-01', periods=n_periods, freq='Y')
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)

print(forecast_df)