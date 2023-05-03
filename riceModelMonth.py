# import module
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import math


# Load data
lab = pd.read_csv('lab.csv') # ข้อมุลแรงงานทางการเกษตร
area = pd.read_excel('ข้าว_พื่นที่ปลูก_ผลผลิต64-57.xlsx')
price = pd.read_excel('ราคาข้าว.xlsx')

# Data preparation

# ใช้ข้อมูลแรงงานตั่งแต่ปี 2557 - 2564
lab = lab.set_index('year')
lab = lab.drop(index = [2554,2555,2556,2565])
lab['รวม'] = lab['รวม'].str.replace(',', '').astype(float)

# นำรวมข้าวเจ้าทุกชนิดรวมทั่วประเทศ
area['Type'].fillna('', inplace=True)
area_filtered = area.loc[area['Type'].str.contains('ข้าวเจ้า')].drop(['City', 'Type'], axis=1) # เลือกชนิดข้าว แก้ contains('ชนิดข้าว')
area_filtered.columns = ['year', 'area_planted', 'area_harvested', 'yield'] #  แก้ชื่อ column เป็น eng
area_filtered = area_filtered.apply(pd.to_numeric, errors='coerce') # เปลี่ยนตัวเลขเป็น int

total_area_planted = area_filtered.groupby('year')['area_planted'].sum() # พื่นที่เพาะปลูก
total_area_harvested = area_filtered.groupby('year')['area_harvested'].sum() # พื่นที่เก็บเกี่ยว
total_yield = area_filtered.groupby('year')['yield'].sum() # ผลผลิตทั่งหมด

# Cobb-Douglas production function
alpha = 0.5  # parameter to be estimated
Y = total_yield
X = pd.DataFrame({'K': total_area_planted, 'N': total_area_harvested ,'M': lab['รวม']})
X = sm.add_constant(X)  # add constant term for estimation

# Fit Cobb-Douglas function with OLS
model = sm.OLS(np.log(Y), np.log(X)).fit()
print(model.summary())

X_scaled = np.log(X) # scale x ลงเพื่อให้ Predict ได้
Y_pred = np.exp(model.predict(X_scaled))

# Calculate R-squared
r_squared = model.rsquared
print("R-squared:", r_squared)

# Calculate Mean Squared Error
mse = ((Y - Y_pred) ** 2).mean()
print("Mean Squared Error:", mse)
print("Root MSE" , math.sqrt(mse) )

# Calculate Mean Absolute Error
mae = np.abs(Y - Y_pred).mean()
print("Mean Absolute Error:", mae)

# Show actual data and predict data
print("Actual yield")
print(total_yield)
print("Predict yield")
print(Y_pred)

# VAR

# import data
export = pd.read_excel('export_month.xlsx', index_col=0)
oil_gas = pd.read_excel('oil_month.xlsx', index_col=0)
price = pd.read_csv('ข้าว.csv')


# data prep
Y_pred = pd.DataFrame(Y_pred).reset_index()
Y_pred = Y_pred.sort_values('year', ascending= False).reset_index().drop(columns='index')
riceEx = export.iloc[1]
riceEx_vol = export.iloc[2]
riceEx = riceEx.iloc[::-1]
riceEx_vol = riceEx_vol.iloc[::-1]
oil = oil_gas.iloc[4]
gas = oil_gas.iloc[10]
oil = oil.iloc[::-1]
gas = gas.iloc[::-1]
data = pd.concat((oil,gas),axis= 1).reset_index()
data2 = pd.concat((riceEx,riceEx_vol),axis = 1).reset_index()
data2 = data2.drop(columns= 'index')
data3 = pd.concat((data.iloc[:-15],data2),axis =1)

# Create a list of all possible combinations of year and month
years = [2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
year_month_combos = [(y, m) for y in years for m in months]

# Create a new data frame with the year and month columns
df_year_month = pd.DataFrame(year_month_combos, columns=['year', 'month'])
data3 = pd.concat((df_year_month,data3),axis=1)
data3 = data3.drop(columns= 'index')
price1 = price.iloc[83860:155034] # use only 2557 - 2564

# group the data by year and month, and calculate the average price
avg_price = pd.DataFrame(price1.groupby(['year', 'month'])['value'].mean())
data3 = data3.reset_index()
avg_price = avg_price.reset_index()
avg_price = avg_price.drop(columns=['year','month'])
all_data = pd.concat((data3,avg_price),axis= 1)
all_data = all_data.drop(columns='index')
all_data = pd.merge(all_data, Y_pred, on='year') # use production from cobb douglas as one of the value
all_data['year_month'] = all_data.apply(lambda row: str(row['year']) + '-' + str(row['month']).zfill(2), axis=1)
all_data = all_data.drop(columns=['year','month']).set_index('year_month')

# Extract the 'value' column
value_col = all_data.pop('value')

# Insert the 'value' column at the last position
all_data.insert(len(all_data.columns), 'value', value_col)
all_data = all_data.apply(pd.to_numeric, errors='coerce').dropna()

# Train-test split
train_size = int(len(all_data) * 0.8)
train_data, test_data = all_data.iloc[:train_size, :], all_data

# Fit VAR model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(train_data)
results = model.fit()

# Make predictions
lag_order = results.k_ar

# Predict the value test on train
var_model = model.fit(lag_order)

prediction = var_model.forecast(all_data.values[-lag_order:], steps=14)
lag_order = results.k_ar
pred = results.forecast(train_data.values[-lag_order:], len(test_data))
pred = pd.DataFrame(pred, index=test_data.index, columns=test_data.columns)

print("Actual Data")
print(all_data)
print("Preict Data(test on train)")
print(pred)

rmse = np.sqrt(mean_squared_error(test_data['value'], pred['value']))
mae = mean_absolute_error(test_data['value'], pred['value'])
r2 = r2_score(test_data['value'], pred['value'])
print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")