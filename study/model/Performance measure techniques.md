### Performance를 측정하는 여러가지 sampling 기법에 대한 개념과, scikit-learn 으로 샐행시키는 방법


https://scikit-learn.org/stable/modules/model_evaluation.html
```
모델 성능 평가할 수 있는 지표에 대한 내용이 설명 되어있는 scikit-learn 공식문서 사이트
```


## Holdout Method (Sampling)
```python3
from sklearn.model_selection import train_test_split
```
```
- 데이터를 Training과 Test로 나누어 모델을 생성하고 테스트 하는기법
- 가장 일반적인 모델 생성을 위한 데이터 랜덤 샘플링 기법
- 비율은 데이터의 크기에 따라 다르게 선택 
```

## Training - Validation - Test
![image](https://user-images.githubusercontent.com/34879309/73653289-33875000-46cc-11ea-89ea-846289bd33a5.png)
```
- Test Set은 Model이 생성시 절대 Training Set에 포함되지 않아야 함
- Test Set과 달리 Model 생성시 Model에 성능을 평가하기 위해 사용
- Hyper Parameter Tuning 시, 성능 평가를 통해 Overfitting 방지
- Training 중간에 Model의 성능을 점검
Tr   V  Tst
 6 : 2 : 2
```

## K-fold cross validation
```
- 학습 데이터를 K번 나눠서 Test와 Train을 실시 -> Test의 평균값을 이용
- 모델의 Parameter 튜닝, 간단한 모델의 최종 성능 측정 등 사용
```
![image](https://user-images.githubusercontent.com/34879309/73653536-b5777900-46cc-11ea-8812-447a84294eb6.png)
