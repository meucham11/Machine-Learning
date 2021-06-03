```python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:03:53 2021

@author: User
"""
import pandas as pd

data = pd.read_csv('D:/practice.csv',encoding='cp949')

#1.
data.info()

#결측치가 있는지 확인하자.
data.isnull().sum()


#2.
data['거래명세서번호']=data['거래명세서번호'].astype('str')

#2-1  , 없애주고 type 바꿔주기
data['원화판매금액계']=data['원화판매금액계'].fillna('0')
data['원화판매금액계']=data['원화판매금액계'].apply(lambda x : x.replace(',','')).astype('int')
data['판매금액']=data['판매금액'].apply(lambda x : x.replace(',','')).astype('float')

data.info()
data


#3-1
data['year1']=data['거래명세서번호'].str[:4]
data['month1']=data['거래명세서번호'].str[4:6]
data['day1']=data['거래명세서번호'].str[6:8]

#3-2
data['date']=pd.to_datetime(data['거래명세서번호'].str[:8])
data['year']=data['date'].dt.year
data['month']=data['date'].dt.month
data['day']=data['date'].dt.day

data.info()



#4. ()와 내용 컬럼 추가
data['bracket']=data['청구처'].str.extract(r'(\(+[\S+]+\))').head(500)

#5.
data.to_csv('D:/reuslt.csv',encoding='cp949')

####################################################################################
# 총 판매금액
round(sum(data['판매금액']),1)
import numpy as np
np.floor(sum(data['판매금액']/1000))*1000
np.ceil(sum(data['판매금액']/1000))*1000

# 평균 판매금액
round(data['판매금액'].mean(),1)
np.floor(data['판매금액'].mean()/0.1)*0.1
np.ceil(data['판매금액'].mean()/0.1)*0.1


# 판매 금액이 가장 큰 품목소분류는?
data[data['판매금액']==max(data['판매금액'])]['품목소분류']

# 가장 거래 빈도가 많은 청구처명과 횟수는?
data['청구처'].value_counts().index[0]
data['청구처'].value_counts()[0]


# 원화판매금액계를 기준으로 오름차순 정리 했을 때 100번째 거래명세서번호는?
data.sort_values('원화판매금액계',ascending=True)['거래명세서번호'][99]


# 판매금액 합이 큰 순서대로 1,2,3위 라고 할 대, 4순위 품목소분류는 무엇인가?
data[['품목소분류','판매금액']].groupby(['품목소분류']).sum().sort_values('판매금액',ascending=False).index[3]
data[['품목소분류','판매금액']].groupby(['품목소분류']).sum().sort_values('판매금액',ascending=False).reset_index(drop=False)['품목소분류'][3]

data[['품목소분류','판매금액']].groupby(['품목소분류']).sum().sort_values('판매금액',ascending=False)['판매금액'][3]


data[['품목소분류','판매금액']].groupby(['품목소분류']).sum().sort_values('판매금액',ascending=False)






```
