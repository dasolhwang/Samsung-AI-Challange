#Model,Data 폴더 생성
#Data폴더 내에 아래 파일 필요
#Kor_Train_교통사망사고정보(12.1~17.6).csv
#result_kor.csv
#test_kor.csv

#아래 코드 실행
import pandas as pd
import Model

train_data = pd.read_csv('./Data/Kor_Train_교통사망사고정보(12.1~17.6).csv',encoding='cp949')
Model.Train(train_data).train()

result = pd.read_csv('./Data/result_kor.csv',encoding='cp949')
test_data = pd.read_csv('./Data/test_kor.csv',encoding='cp949')
row = result['행'].values - 2
col = result['열'].values

result['값'] = Model.Predict(row, col, test_data).predict()
result.to_csv('result.csv',index=False)