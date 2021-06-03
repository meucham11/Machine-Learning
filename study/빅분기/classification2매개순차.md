```python3
# https://www.kaggle.com/lifesailor/xgboost


import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve
import xgboost as xgb

import warnings
warnings.filterwarnings(action='ignore') 

# 데이터 로드
train_x_raw = pd.read_csv('D:\\jupyter lab\\빅분기\\작업2유형/X_train.csv',encoding='cp949')
train_y_raw = pd.read_csv('D:\\jupyter lab\\빅분기\\작업2유형/Y_train.csv',encoding='cp949')
test_x_raw = pd.read_csv('D:\\jupyter lab\\빅분기\\작업2유형/X_test.csv',encoding='cp949')

train_x = train_x_raw.copy()
train_y = train_y_raw.copy()
test_x = test_x_raw.copy()

############################### eda
## info
train_x.info()
train_y.info()

## 궁금한 컬럼 살펴보기
train_x['내점일수']

# # train_y의 gender 가 0,1 이 아니라 m,f 이라면
'''
yn={'m':0,
    'f':1}
data['gender']=data.replace({'gender':yn})['gender']
'''


## object인 변수들의 고유값이 몇개 있는지 파악
len(train_x['주구매상품'].unique()) #42
len(train_x['주구매지점'].unique()) #24


## 시각화가 되지 않기 때문에 데이터 분포 살필 수 없음


############################### feature engineering
## 필요없는 열 삭제
del train_x['cust_id']
del test_x['cust_id']
del train_y['cust_id']

## 결측치 탐색은 하되, xgboost 에서 use_missing=False 를 활용할 것임
train_x.iloc[:,1:].isnull().sum()
sum(train_x['환불금액']==0)### 환불금액이 nan인건 0으로 대체해도 될듯하다.
train_x['환불금액']=train_x['환불금액'].fillna(0)
test_x['환불금액']=test_x['환불금액'].fillna(0)



###############################################################





###############################################################

## xgboost에서는 scaling을 하지 않아도 큰 성능 차이가 없으므로 패스

## 더미화
train_x = pd.get_dummies(train_x,columns=['주구매상품','주구매지점'])
test_x = pd.get_dummies(test_x,columns=['주구매상품','주구매지점'])
# 여기서 문제 발생 train 더미화와 test 더미화했을 때 컬럼명이 매칭되지 않는것이 있다.
list(set(train_x)-set(test_x))
del train_x[list(set(train_x)-set(test_x))[0]]

############################### split  train 데이터를 다시 train과 test로 나눈다.
x_train, x_test, y_train, y_test = train_test_split(train_x,train_y, random_state=1)

## 1. max_depth와 min_child_weight를 튜닝한다.
param_test1 = {
 'max_depth':range(3,10,3),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate=0.1, 
                                                  n_estimators=1000, 
                                                  max_depth=5, 
                                                  min_child_weight=1, 
                                                  gamma=0, 
                                                  subsample=0.8, 
                                                  colsample_bytree=0.8,
                                                  objective= 'binary:logistic', 
                                                  nthread=-1, 
                                                  scale_pos_weight=1, seed=2019),
param_grid = param_test1, scoring='accuracy',n_jobs=-1,iid=False, cv=5, verbose=10)
gsearch1.fit(x_train,y_train)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


## 2. Gamma를 튜닝한다.
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, 
                                                  n_estimators=1000, 
                                                  max_depth=gsearch1.best_params_['max_depth'],
                                                  min_child_weight=gsearch1.best_params_['min_child_weight'], 
                                                  gamma=0, 
                                                  subsample=0.8, 
                                                  colsample_bytree=0.8,
                                                  objective= 'binary:logistic', 
                                                  thread=-1, 
                                                  scale_pos_weight=1,
                                                  seed=2019), 
                        param_grid = param_test2, scoring='accuracy', n_jobs=-1, iid=False, cv=5)
gsearch2.fit(x_train,y_train)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


## 3. subsample and colsample_bytree를 튜닝한다.
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, 
                                                  n_estimators=1000, 
                                                  max_depth=gsearch1.best_params_['max_depth'],
                                                  min_child_weight=gsearch1.best_params_['min_child_weight'], 
                                                  gamma=gsearch2.best_params_['gamma'], 
                                                  subsample=0.8, 
                                                  colsample_bytree=0.8,
                                                  objective= 'binary:logistic', 
                                                  thread=-1, 
                                                  scale_pos_weight=1,
                                                  seed=2019), 
                        param_grid = param_test3, scoring='accuracy', n_jobs=-1, iid=False, cv=5, verbose=10)
gsearch3.fit(x_train,y_train)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


## 4. subsample 추가 튜닝한다.
param_test4 = {
 'subsample':[i/100.0 for i in range(40,80)],
}
gsearch4 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, 
                                                  n_estimators=1000, 
                                                  max_depth=gsearch1.best_params_['max_depth'],
                                                  min_child_weight=gsearch1.best_params_['min_child_weight'], 
                                                  gamma=gsearch2.best_params_['gamma'], 
                                                  subsample=gsearch3.best_params_['subsample'], 
                                                  colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                                                  objective= 'binary:logistic', 
                                                  thread=-1, 
                                                  scale_pos_weight=1,
                                                  seed=2019), 
                        param_grid = param_test4, scoring='accuracy', n_jobs=-1, iid=False, cv=5, verbose=10)
gsearch4.fit(x_train,y_train)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_



## 5. Regularization Parameter 튜닝
param_test5 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch5 = GridSearchCV(estimator = xgb.XGBClassifier(learning_rate =0.1, 
                                                  n_estimators=1000, 
                                                  max_depth=gsearch1.best_params_['max_depth'],
                                                  min_child_weight=gsearch1.best_params_['min_child_weight'], 
                                                  gamma=gsearch2.best_params_['gamma'], 
                                                  subsample=gsearch4.best_params_['subsample'], 
                                                  colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                                                  objective= 'binary:logistic', 
                                                  thread=-1, 
                                                  scale_pos_weight=1,
                                                  seed=2019), 
                        param_grid = param_test5, scoring='accuracy', n_jobs=-1, iid=False, cv=5, verbose=10)
gsearch5.fit(x_train,y_train)
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_


## 6. learning rate 감소

# Evaluate Optimized Model
clf_xgb = xgb.XGBClassifier(learning_rate =0.1, 
                            n_estimators=1000, 
                            max_depth=gsearch1.best_params_['max_depth'],
                            min_child_weight=gsearch1.best_params_['min_child_weight'], 
                            gamma=gsearch2.best_params_['gamma'], 
                            subsample=gsearch4.best_params_['subsample'], 
                            colsample_bytree=gsearch3.best_params_['colsample_bytree'],
                            objective= 'binary:logistic', 
                            thread=-1, 
                            scale_pos_weight=1,
                            seed=2019
                        )
clf_xgb.fit(x_train, 
            y_train, 
            verbose=True, 
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(x_test, y_test)])



# train 모델 평가
preds = clf_xgb.predict(x_test)

prob=clf_xgb.predict_proba(x_test)
prob
accuracy = (preds.flatten() == np.array(y_test).flatten()).sum().astype(float) / len(preds)*100
accuracy



####################################################################################################
# test 데이터 대입
preds = clf_xgb.predict(test_x)
preds
prob=clf_xgb.predict_proba(test_x)



## csv 생성

list(zip(*prob))[1]

submit_csv = pd.DataFrame({'custid':test_x_raw['cust_id'],
                           'gender':list(zip(*prob))[1]})
submit_csv.to_csv('D:\\jupyter lab\\빅분기\\작업2유형/submit_csv',encoding='cp949')
```
