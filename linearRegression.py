"""
Author: Xuchen Sun
license: General Public License （GPL）
Contact: xuchens@mun.ca
Hardware Note CPU:AMD3900
Hardware Note GPU: EVGA GTX1080ti
Date: 2021-08-09
LinearRegression by Scikit_learn
"""
import warnings
from sklearn.linear_model import LinearRegression



warnings.filterwarnings('ignore')
model = LinearRegression()  # use linearRegression Model



"train the model with three point[0,0],[1,1],[2,2] and the last array [1,2,3] means three value of the previvous three points"
model.fit([[0, 0], [1, 1], [2, 2]], [1, 2, 3])

print(model.coef_, model.intercept_)


result_of_predict=model.predict([[3, 3]])

print(result_of_predict)
