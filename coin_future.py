import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# CSV 파일에서 코인 가격 데이터 불러오기
df = pd.read_csv('upbit_bitcoin_prices.csv')

# 'Date' 컬럼을 날짜 형식으로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 데이터를 날짜 기준으로 오름차순 정렬 (가장 최신 데이터가 마지막에 오도록)
df = df.sort_values('Date')

# 'Date' 컬럼을 인덱스로 설정
df.set_index('Date', inplace=True)

# 종가(Close)만 사용
data = df['Close'].values
data = data.reshape(-1, 1)

# 데이터 스케일링 (LSTM은 데이터 스케일링이 필요함)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 데이터셋을 학습용과 테스트용으로 나누기 (70% 학습, 30% 테스트)
train_size = int(len(scaled_data) * 0.7)  # 학습 데이터를 70%로 설정
test_size = len(scaled_data) - train_size  # 나머지 30%를 테스트 데이터로 설정
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 시퀀스 데이터 생성 (과거 200일을 기준으로 예측)
time_step = 200  # 여기서 200을 사용하고자 하신다면, 훈련 데이터가 충분한지 확인

# train_data와 test_data가 충분한 길이를 가지는지 확인하고, X_train, X_test 생성
if len(train_data) <= time_step:
    print(f"Train data is too small for the specified time_step ({time_step}). Reducing time_step to {min(len(train_data)-1, 30)}.")
    time_step = min(len(train_data)-1, 30)  # 훈련 데이터 길이에 맞춰 time_step을 줄입니다.

# create_dataset 함수 정의
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# train_data와 test_data가 충분한 길이를 가지는지 확인하고, X_train, X_test 생성
if len(train_data) > time_step:
    X_train, y_train = create_dataset(train_data, time_step)
else:
    raise ValueError("Train data is too small for the specified time_step.")

if len(test_data) > time_step:
    X_test, y_test = create_dataset(test_data, time_step)
else:
    # time_step을 더 작은 값으로 조정하여 해결할 수 있습니다.
    print(f"Test data is too small for the specified time_step. Reducing time_step to {min(len(test_data), 30)}")
    time_step = min(len(test_data), 30)
    X_test, y_test = create_dataset(test_data, time_step)

# LSTM 모델의 입력 형태에 맞게 데이터 형태 변경 (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM 모델 정의
model = Sequential()

# LSTM 레이어 추가
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# 출력 레이어 (Dense 레이어)
model.add(Dense(units=1))

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 예측 및 시각화
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측 데이터를 원래 스케일로 되돌리기
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# 실제 값 되돌리기
train_data_actual = scaler.inverse_transform(train_data)
test_data_actual = scaler.inverse_transform(test_data)

# 예측 결과 시각화
plt.figure(figsize=(12, 6))

# 실제 데이터 (test data) 시각화
plt.plot(df.index[train_size + time_step:], test_data_actual[time_step:], color='blue', label='Real Bitcoin Price')

# test_predict의 마지막 값 제거하여 길이 맞추기
test_predict = test_predict[:len(test_data_actual[time_step:])]  # 실제 test_data의 길이에 맞게 크기 조정

# 예측 결과 (test predict) 시각화
plt.plot(df.index[train_size + time_step:][-len(test_predict):], test_predict, color='red', label='Predicted Bitcoin Price')

plt.title('Bitcoin Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.show()
