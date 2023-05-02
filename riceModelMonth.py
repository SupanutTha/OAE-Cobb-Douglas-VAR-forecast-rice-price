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


# data prep
Y_pred = pd.DataFrame(Y_pred).reset_index()
Y_pred = Y_pred.sort_values('year', ascending= False).reset_index().drop(columns='index')
riceEx = export.iloc[1]
riceEx_vol = export.iloc[2]
riceEx = riceEx.iloc[::-1]
riceEx_vol = riceEx_vol.iloc[::-1]
oil = oil_gas.iloc[1]
gas = oil_gas.iloc[10]
oil = oil.iloc[::-1]
gas = gas.iloc[::-1]
data = pd.concat((oil,gas),axis= 1).reset_index()
data2 = pd.concat((riceEx,riceEx_vol),axis = 1).reset_index()
data2 = data2.drop(columns= 'index')

print(riceEx)
print(riceEx_vol)
print(oil)
print(gas)
print(data.iloc[:-15])
print(data2)
data3 = pd.concat((data.iloc[:-15],data2),axis =1)
print(data3.set_index('index'))



print(data3)