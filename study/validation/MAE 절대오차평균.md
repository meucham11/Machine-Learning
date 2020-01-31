# MAE
```
mean absolute error
절대 오차의 평균
```
![mae](https://user-images.githubusercontent.com/34879309/73507688-1a0cac80-441d-11ea-8cf7-0c0cc8e371b7.gif)
```
방향을 고려하지 않고 일련의 예측에서 오류의 평균 크기를 측정한다.
오차 계산시 양수 및 음수 오류가 취소되므로 주의해야한다.

값이 낮을수록 좋다.
분산이 클 때는 RMSE보다 안정적이다. -> RMSE가 큰 오류를 더 반영할 수 있다는 뜻.
```


```python3
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```

