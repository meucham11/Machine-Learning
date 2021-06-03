
```python3
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


## xgboost에서는 scaling을 하지 않아도 큰 성능 차이가 없으므로 패스

## 더미화
train_x = pd.get_dummies(train_x,columns=['주구매상품','주구매지점'])
test_x = pd.get_dummies(test_x,columns=['주구매상품','주구매지점'])
# 여기서 문제 발생 train 더미화와 test 더미화했을 때 컬럼명이 매칭되지 않는것이 있다.
list(set(train_x)-set(test_x))
del train_x[list(set(train_x)-set(test_x))[0]]

############################### split  train 데이터를 다시 train과 test로 나눈다.
x_train, x_test, y_train, y_test = train_test_split(train_x,train_y, random_state=1)


param_grid = {
    'max_depth': [2,3, 4],
    'learning_rate': [0.5,0.25,0.1,0.05,0.01],
    'gamma': [0,0.2, 0.25,0.3, 0.5],
    'reg_lambda': [1,5,10,12],
    'scale_pos_weight': [1.5, 3, 4] # NOTE: XGBoost recommends sum(negative instances) / sum(positive instances)
}
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', 
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5,
                                use_label_encoder=False),
    
    param_grid=param_grid,
    scoring='roc_auc', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0, # NOTE: If you want to see what Grid Search is doing, set verbose=2
    n_jobs = 10,
    cv = 10
)

optimal_params.fit(x_train, 
                   y_train, 
                   early_stopping_rounds=10,                
                   eval_metric='auc',
                   eval_set=[(x_test, y_test)],
                   verbose=False)
print(optimal_params.best_params_)


# Evaluate Optimized Model
clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        subsample=0.9,
                        colsample_bytree=0.5,
                        max_depth=optimal_params.best_params_['max_depth'],
                        learning_rate=optimal_params.best_params_['learning_rate'],
                        gamma=optimal_params.best_params_['gamma'],                        
                        reg_lambda=optimal_params.best_params_['reg_lambda'],
                        scale_pos_weight=optimal_params.best_params_['scale_pos_weight'],
                        use_label_encoder=False
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
'''
