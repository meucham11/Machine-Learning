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
```
from sklearn.model_selection import cross_validation
에서 scores를 뽑았을 때 scoring = 'neg_mean_squared_error'를 입력함으로써 -값이 붙어서 나온다.
이는 scikit learn의 정책, 즉, rmse나 mse는 작아야 좋은데 score값은 높으면 좋으므로 헷갈리지 않게 모든 지표값을 작아야 좋다는 것으로 통일하기 위함이다.

```

## Leave One Out (LOO)
```
 - 거의 잘 안쓰는 기법이다.
 - =simple cross validation으로 불리며 한번에 한 개의 데이터만 Test Set으로 사용한다 -> 총 k번의 iteration
 - 파라메타가 없다.
```
---
# Check variation of cross validation
### lasso scores /  ridge scores
<img width="247" alt="캡처" src="https://user-images.githubusercontent.com/34879309/73716040-472cc800-4759-11ea-8aad-2ab656a70534.PNG">
```
ridge 모델이 데이터에 더 민감하게 만들어진다.(분포가 넓기 때문에)
이 그림에서는 최대 모델은 릿지가 좋지만
안정성으로 보았을 때는 라쏘가 더 낫다는 평
```
---
# Validation set for parameter tuning
```
 - Validation set의 많은 이유중 하나가 Hyper parameter tuning이다.
 - Number of iterations(SGD), Number of branch(Tree-based) 등등
 - Validation set의 성능으로 최적의 parameter를 찾음
 - Validation set 결과와 Training set 겨ㄹ과의 차이가 벌어지면 overfitting
```

