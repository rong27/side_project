'''
L45 製程參數 平均長度 
平均長度資料中，沒有離群值
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
from scipy.stats import normaltest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import sqrt

data = pd.read_csv(r".\data\L45_parameters_averageLength.csv")

print(data.info())

# averageLength 五數彙總、boxplot
dataaverageLength = data['averageLength'].describe()

print(dataaverageLength)
averageLengthP25 = dataaverageLength['25%']
averageLengthP75 = dataaverageLength['75%']
print(averageLengthP75)
print(averageLengthP25)
IQR = averageLengthP75-averageLengthP25
print(IQR)
upper_boundary_limit_averageLength = round(averageLengthP75 + 1.5*IQR, 3)
lower_boundary_limit_averageLength = round(averageLengthP25 - 1.5*IQR, 3)
e_upper_boundary_limit_averageLength = round(averageLengthP75 + 3*IQR, 3)
e_lower_boundary_limit_averageLength = round(averageLengthP25 - 3*IQR, 3)
extrem_normal_interval = [e_lower_boundary_limit_averageLength, e_upper_boundary_limit_averageLength]
normal_interval = [lower_boundary_limit_averageLength, upper_boundary_limit_averageLength]
print(f'Extreme normal averageLength interval : {extrem_normal_interval} ')
print(f'Normal averageLength interval : {normal_interval} ')

# boxplot
plt.boxplot(data['averageLength'],
            patch_artist = True, # 把盒子填色，預設是白色
            boxprops = {'color':'#4169E1', 'facecolor':'#6495ED'},
            flierprops= {'marker':'o', 'markerfacecolor':'#87CEFA', 'markeredgecolor':'#4169E1'},
            medianprops= {'linestyle':'--', 'color':'#4B0082', 'linewidth':'1'},
            capprops = {'color':'#4169E1'}, # min max 線段屬性
            whiskerprops= {'color':'#4169E1', 'linestyle':'--'})
plt.title('Average Length Boxplot')
# plt.xlabel('')
plt.ylabel('Average Length (mm)')
plt.show()
#
# outlier index
outlierIndex = []
for outlierI, outlierV in enumerate(data['averageLength']):
    if outlierV < normal_interval[0] or outlierV > normal_interval[1]:
        print(outlierI, ':', outlierV)
        outlierIndex.append(outlierI)
print(outlierIndex)
drop_outliers = data.drop(outlierIndex)
print('Before remove outliers shape:', data.shape)
print('After remove outliers shape:', drop_outliers.shape)

# # After remove outliers 
# Decide input(x) and output(y)
x = drop_outliers.iloc[:, 1:12].values # 去掉 no 這欄
y = drop_outliers.iloc[:, 12].values 


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Feature extraction 

## 決策樹 77-180
decisionTreeModel = DecisionTreeRegressor(criterion='squared_error',
                                          max_depth=5,
                                          random_state=42)


decisionTreeModel.fit(x_train, y_train)
predicted_train = decisionTreeModel.predict(x_train)
predicted_test = decisionTreeModel.predict(x_test)

def mape(y_true, y_predict):
    return np.mean(np.abs((y_predict - y_true) / y_true)) * 100


print('------------------Training data------------------')
R2_train = decisionTreeModel.score(x_train, y_train)
mse_train = metrics.mean_squared_error(y_train, predicted_train)
mae_train = metrics.mean_absolute_error(y_train, predicted_train)

print('R2 score(train): ', round(R2_train, 3))
print('MAE(train) : ', round(mae_train, 3))
print('MSE(train): ', round(mse_train, 3))
print(f'RMSE (training_data) : {round(sqrt(mse_train), 3)}')

print('------------------Testing data------------------')
R2_test = decisionTreeModel.score(x_test, y_test)
mse_test = metrics.mean_squared_error(y_test, predicted_test)
mae_test = metrics.mean_absolute_error(y_test, predicted_test)

print('R2 score(test): ', round(R2_test, 3))
print('MAE(test) : ', round(mae_test, 3))
print('MSE(test): ', round(mse_test, 3))
print('RMSE(test): ', round((sqrt(mse_test)), 3))


## 特徵重要性
importances = decisionTreeModel.feature_importances_
i_num = []
for i in importances:
    i_num.append(round(i, 3))
f_name = []
for f in data.columns[1:12]:
    f_name.append(f)

importances_reversed = [num for num in reversed(np.argsort(i_num))]

print('------------------Features Ranked------------------')
new = list(zip(f_name, i_num))
initail_rank = 1
ranked_f = []
ranked_i = []
for index in importances_reversed:
    print(f'{initail_rank}) {new[index][0]} : {new[index][1]}')
    ranked_f.append(new[index][0])
    ranked_i.append(new[index][1])
    initail_rank += 1

## 特徵重要性圖(排序)
plt.barh(ranked_f, ranked_i, color='navy')
for a, b in enumerate(ranked_i):
    # print(f'a:{a}, b:{b}')
    plt.text(b, ranked_f[a], str(f' {b}'), color='navy', fontsize=14)
# plt.tick_params(axis='x', labelsize=5)
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.yticks(fontsize=14)
plt.title('Average Length in Decision Tree-Feature importances', fontsize=14)
plt.show()

## 散佈圖 (訓練集)
x_axis = range(1, len(x_train)+1)
plt.plot(x_axis, y_train, 'o', color='lightcoral', label='True', markersize=8)
plt.plot(x_axis, predicted_train, 'o', color='navy', label='Predict(train)', markersize=6)
plt.title('Decision Tree-Average Length prediction (train)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

## 折線圖 (訓練集)
plt.plot(x_axis, y_train, color='lightcoral', label='True', linewidth=4)
plt.plot(x_axis, predicted_train, color='navy', label='Predict(train)', linewidth=2)
plt.title('Decision Tree-Average Length prediction (train)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

## 散佈圖 (訓練集)
x_axis_test = range(1, len(x_test)+1)
plt.plot(x_axis_test, y_test, 'o', color='lightcoral', label='True', markersize=8)
plt.plot(x_axis_test, predicted_test, 'o', color='navy', label='Predict(test)', markersize=6)
plt.title('Decision Tree-Average Length prediction (test)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend(loc='best')
plt.show()
## 折線圖 (測試集)
plt.plot(x_axis_test, y_test, color='lightcoral', label='True', linewidth=4)
plt.plot(x_axis_test, predicted_test, color='navy', label='Predict(test)', linewidth=2)
plt.title('Decision Tree-Average Length prediction (test)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend(loc='best')
plt.show()


# RF 183-284
randomForestModel = RandomForestRegressor(n_estimators=25,
                                          criterion='squared_error',
                                          random_state=42)

randomForestModel.fit(x_train, y_train)

predicted_train = randomForestModel.predict(x_train)
predicted_test = randomForestModel.predict(x_test)

print('------------------Training data------------------')
R2_train = randomForestModel.score(x_train, y_train)
mse_train = metrics.mean_squared_error(y_train, predicted_train)
mae_train = metrics.mean_absolute_error(y_train, predicted_train)

print('R2 score(train): ', round(R2_train, 3))
print('MAE(train) : ', round(mae_train, 3))
print('MSE(train): ', round(mse_train, 3))
print(f'RMSE (training_data) : {round(sqrt(mse_train), 3)}')

print('------------------Testing data------------------')
R2_test = randomForestModel.score(x_test, y_test)
mse_test = metrics.mean_squared_error(y_test, predicted_test)
mae_test = metrics.mean_absolute_error(y_test, predicted_test)

print('R2 score(test): ', round(R2_test, 3))
print('MAE(test) : ', round(mae_test, 3))
print('MSE(test): ', round(mse_test, 3))
print('RMSE(test): ', round((sqrt(mse_test)), 3))

print('------------------Features Ranked------------------')
importances = randomForestModel.feature_importances_
# print(importances)
importances_reversed = [num for num in reversed(np.argsort(importances))]
# print(importances_reversed)

round_importances = []
for round_i in importances:
    round_importances.append(round(round_i, 3))
# print(f'特徵重要性(四捨五入) : {round_importances}')

f_name = []
for f in data.columns[1:12]:
    f_name.append(f)

new = list(zip(f_name, round_importances))
initail_rank = 1
ranked_f = []
ranked_i = []
for index in importances_reversed:
    print(f'{initail_rank}) {new[index][0]} : {new[index][1]}')
    ranked_f.append(new[index][0])
    ranked_i.append(new[index][1])
    initail_rank += 1

# Ranked
plt.barh(ranked_f, ranked_i, color='navy')
for a, b in enumerate(ranked_i):
    # print(f'a:{a}, b:{b}')
    plt.text(b, ranked_f[a], str(f' {b}'), color='navy', fontsize=14)
# plt.tick_params(axis='x', labelsize=5)
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.yticks(fontsize=14)
plt.title('Average Length in Random Forest-Feature importances', fontsize=14)
plt.show()


# predict graph
x_axis = range(1, len(x_train)+1)
plt.plot(x_axis, y_train, 'o', color='lightcoral', label='True')
plt.plot(x_axis, predicted_train, 'o', color='navy', label='Predict(train)')
plt.title('Random Forest-Average Length prediction (train)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

plt.plot(x_axis, y_train, color='lightcoral', label='True', linewidth=4)
plt.plot(x_axis, predicted_train, color='navy', label='Predict(train)', linewidth=2)
plt.title('Random Forest-Average Length prediction (train)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()


x_axis_test = range(1, len(x_test)+1)
plt.plot(x_axis_test, y_test, 'o', color='lightcoral', label='True')
plt.plot(x_axis_test, predicted_test, 'o', color='navy', label='Predict(test)')
plt.title('Random Forest-Average Length prediction (test)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

plt.plot(x_axis_test, y_test, color='lightcoral', label='True', linewidth=4)
plt.plot(x_axis_test, predicted_test, color='navy', label='Predict(test)', linewidth=2)
plt.title('Random Forest-Average Length prediction (test)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

# XGBoost 286-
xgbrModel = xgb.XGBRegressor(
    n_estimators=40,
    max_depth=5,
    booster='gbtree',
    learning_rate=0.3
)

xgbrModel.fit(x_train, y_train)

predicted_train = xgbrModel.predict(x_train)
predicted_test = xgbrModel.predict(x_test)

print('------------------Training data------------------')
R2_train = xgbrModel.score(x_train, y_train)
mse_train = metrics.mean_squared_error(y_train, predicted_train)
mae_train = metrics.mean_absolute_error(y_train, predicted_train)
print('R2 score(train): ', round(R2_train, 3))
print('MAE(train) : ', round(mae_train, 3))
print('MSE(train): ', round(mse_train, 3))
print(f'RMSE (training_data) : {round(sqrt(mse_train), 3)}')

print('------------------Testing data------------------')
R2_test = xgbrModel.score(x_test, y_test)
mse_test = metrics.mean_squared_error(y_test, predicted_test)
mae_test = metrics.mean_absolute_error(y_test, predicted_test)
print('R2 score(test): ', round(R2_test, 3))
print('MAE(test) : ', round(mae_test, 3))
print('MSE(test): ', round(mse_test, 3))
print('RMSE(test): ', round((sqrt(mse_test)), 3))


x_axis = range(1, len(x_train)+1)
plt.plot(x_axis, y_train, 'o', color='lightcoral', label='True', markersize=8)
plt.plot(x_axis, predicted_train, 'o', color='navy', label='Predict(train)', markersize=6)
plt.title('XGBoost-Average Length prediction (train)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

plt.plot(x_axis, y_train, color='lightcoral', label='True', markersize=8, linewidth=4)
plt.plot(x_axis, predicted_train, color='navy', label='Predict(train)', linewidth=2)
plt.title('XGBoost-Average Length prediction (train)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend()
plt.show()

x_axis_test = range(1, len(x_test)+1)
plt.plot(x_axis_test, y_test, 'o', color='lightcoral', label='True')
plt.plot(x_axis_test, predicted_test,'o', color='navy', label='Predict(test)')
plt.title('XGBoost-Average Length prediction (test)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend(loc='best')
plt.show()

plt.plot(x_axis_test, y_test, color='lightcoral', label='True',linewidth=4)
plt.plot(x_axis_test, predicted_test, color='navy', label='Predict(test)', linewidth=2)
plt.title('XGBoost-Average Length prediction (test)')
plt.xlabel('No')
plt.ylabel('Average Length (mm)')
plt.legend(loc='best')
plt.show()
#
print('------------------Features Ranked------------------')
importances = xgbrModel.feature_importances_
importances_reversed = [num for num in reversed(np.argsort(importances))]

round_importances = []
for round_i in importances:
    round_importances.append(round(round_i, 3))
# print(f'特徵重要性(四捨五入) : {round_importances}')

f_name = []
for f in data.columns[1:12]:
    f_name.append(f)

new = list(zip(f_name, round_importances))
initail_rank = 1
ranked_f = []
ranked_i = []
for index in importances_reversed:
    print(f'{initail_rank}) {new[index][0]} :', new[index][1])
    ranked_f.append(new[index][0])
    ranked_i.append(new[index][1])
    initail_rank += 1

# Ranked
plt.barh(ranked_f, ranked_i, color='navy')
for a, b in enumerate(ranked_i):
    # print(f'a:{a}, b:{b}')
    plt.text(b, ranked_f[a], str(b), color='navy', fontsize=14)
# plt.tick_params(axis='x', labelsize=5)
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.yticks(fontsize=14)
plt.title('Average Length in XGBoost-Feature importances', fontsize=14)
plt.show()

