import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from prophet import Prophet

# 1단계: 업비트API로 가격 저장하기
# 업비트 시세 조회 API URL (코인 가격 가져오기)
url = "https://api.upbit.com/v1/candles/days"

# API 요청에 필요한 매개변수 설정
def get_bitcoin_data(count=200, to=""):
    params = {
        "market": "KRW-BTC",  # 가져올 코인 선택
        "count": count,  # 가져올 데이터 수 (최대 200)
        "to": to,  # 가져올 종료 시간 
    }
    response = requests.get(url, params=params)
    return response.json()

# 데이터를 가져올 기간 설정
count = 200  # 200일의 데이터

# 가격 데이터 가져오기
data = get_bitcoin_data(count=count)

# 데이터 확인
if data:
    # JSON 응답에서 가격 데이터 추출
    df = pd.DataFrame(data)
    
    # 타임스탬프를 날짜로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 필요한 열만 선택 (timestamp, 가격, 거래량 등)
    df = df[['timestamp', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
    
    # 열 이름 변경 (좀 더 직관적으로)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # CSV 파일로 저장
    df.to_csv("upbit_bitcoin_prices.csv", index=False)
    
    print("코인 가격 데이터를 'upbit_bitcoin_prices.csv' 파일로 저장했습니다.")
else:
    print("데이터를 가져오는 데 실패했습니다.")

# 2단계: CSV에서 데이터를 가져와서 시각화하기

# CSV 파일에서 코인 가격 데이터 불러오기
df = pd.read_csv('upbit_bitcoin_prices.csv')

# DataFrame의 컬럼 확인 (확인용)
print(df.columns)

# 'Date' 컬럼을 날짜 형식으로 변환 (만약 'timestamp' 컬럼이라면)
df['Date'] = pd.to_datetime(df['Date'])  # 만약 'timestamp' 컬럼이라면 df['timestamp']로 변경
df.set_index('Date', inplace=True)

# 코인 가격 시각화: 종가(Close) 그래프
plt.figure(figsize=(10, 6))  # 그래프 크기 설정
plt.plot(df.index, df['Close'], label='Close Price (KRW)', color='blue')  # Date 컬럼을 인덱스로 사용
plt.title('Upbit Bitcoin Price (KRW)', fontsize=14)  # 그래프 제목
plt.xlabel('Date', fontsize=12)  # x축 레이블
plt.ylabel('Price (KRW)', fontsize=12)  # y축 레이블
plt.grid(True)  # 격자 표시
plt.xticks(rotation=45)  # x축 레이블 회전
plt.tight_layout()  # 그래프의 레이아웃을 자동으로 조정
plt.legend()  # 범례 표시

# 그래프 표시
plt.show()

# 3단계: 가격 예측하기
