"""
Author: Xuchen Sun
license: General Public License （GPL）
Contact: xuchens@mun.ca
Hardware Note CPU:AMD3900
Hardware Note GPU: EVGA GTX1080ti
Date: 2021-08-09
3 Steps to predict diabetes
step1: prepare dataset
step2: load linear regression model and train by using previous dataset
step3: predict diabetes by test dataset and draw pictures
"""
from sklearn.linear_model import LinearRegression
from sklearn import datasets  #import dataset
from sklearn.model_selection import train_test_split  # import split model
import numpy as np  # import calculation model
import matplotlib.pyplot as plt  # load matplotlib to draw picture

"Step1:prepare dataset"
diabetes = datasets.load_diabetes()  # load diabetes dataset

diabetes_feature = diabetes.data[:, np.newaxis, 2]  # only use one of feature
diabetes_target = diabetes.target  # set the value for linear Regression


# 80% dataset for train and 20% dataset for test set
# set random_state
train_feature, test_feature, train_target, test_target = train_test_split(diabetes_feature, diabetes_target, test_size=0.2, random_state=50)


"Step2:build model and train it"
model = LinearRegression()
model.fit(train_feature, train_target)



"Step3: predict values by using test dataset and draw images"

# draw pictures
plt.scatter(train_feature, train_target,  color='black')  # train set
plt.scatter(test_feature, test_target,  color='red')  # test set
plt.plot(test_feature, model.predict(test_feature),color='green', linewidth=5)  # draw fitting line

# draw pictures
plt.legend(('Fit line', 'Train Set', 'Test Set'), loc='lower right')
plt.title('LinearRegression(sklearn.dataset.diabetes)')
plt.show()