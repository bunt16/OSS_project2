import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np


def sort_dataset(dataset_df):
  sorted_dataset = dataset_df.sort_values(by='year')#.reset_index(drop=True)
  return sorted_dataset

def split_dataset(dataset_df):
  train = dataset_df.loc[:1718]
  test = dataset_df.loc[1718:]

  X_train = train.drop('salary', axis=1)
  Y_train = train['salary'] * 0.001
  X_test = test.drop('salary', axis=1)
  Y_test = test['salary'] * 0.001

  return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
  dataset_df = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',
'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]
  # NaN 값을 0으로 보간
  # dataset_df = dataset_df.fillna(0)
  # dataset_df = dataset_df.select_dtypes(include='number')

  return dataset_df

def train_predict_decision_tree(X_train, Y_train, X_test):
  # Decision Tree 모델 생성
  model = DecisionTreeRegressor(random_state=42)

  # 모델 훈련
  model.fit(X_train, Y_train)

  # 테스트 데이터에 대한 예측
  predictions = model.predict(X_test)

  return predictions

def train_predict_random_forest(X_train, Y_train, X_test):
	# Random Forest 모델 생성
  model = RandomForestRegressor(random_state=42)

  # 모델 훈련
  model.fit(X_train, Y_train)

  # 테스트 데이터에 대한 예측
  predictions = model.predict(X_test)

  return predictions

def train_predict_svm(X_train, Y_train, X_test):
  # StandardScaler를 사용하여 데이터 스케일링
  scaler = StandardScaler()

  # X_train 데이터 스케일링
  X_train_scaled = scaler.fit_transform(X_train)

  # SVM 모델 생성
  svm_model = SVR()

  # 모델 훈련
  svm_model.fit(X_train_scaled, Y_train)

  # X_test 데이터 스케일링
  X_test_scaled = scaler.transform(X_test)

  # 테스트 데이터에 대한 예측
  predictions = svm_model.predict(X_test_scaled)

  return predictions

def calculate_RMSE(labels, predictions):
    # 예측값이 None이 아닌 경우에만 계산 수행
    if predictions is not None:
        rmse = np.sqrt(mean_squared_error(labels, predictions))
        return rmse
    else:
        return None

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

	sorted_df = sort_dataset(data_df)
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)

	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))