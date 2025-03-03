import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta


# CSV 파일 읽기
df = pd.read_csv('upbit_bitcoin_prices.csv')

# 'Date' 컬럼을 datetime 형식으로 변환하고 인덱스 설정
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 데이터 확인 (일부)
print(df.head())


""" 1단계 기술적 지표 계산"""

# 이동 평균 계산
df['SMA50'] = df['Close'].rolling(window=50).mean()  # 50일 단기 이동 평균
df['SMA200'] = df['Close'].rolling(window=200).mean()  # 200일 장기 이동 평균
# RSI 계산 (21일 기준)
df['RSI'] = ta.rsi(df['Close'], length=21)
# MACD 계산
df['MACD'], df['Signal_Line'], _ = ta.macd(df['Close'], fast=12, slow=26, signal=9)


""" 2단계 매수 매도 신호 생성 """

# 골든 크로스 및 데드 크로스
df['Buy_Signal_MA'] = (df['SMA50'] > df['SMA200']) & (df['SMA50'].shift(1) <= df['SMA200'].shift(1))
df['Sell_Signal_MA'] = (df['SMA50'] < df['SMA200']) & (df['SMA50'].shift(1) >= df['SMA200'].shift(1))
# RSI 기반 매수/매도 신호
df['Buy_Signal_RSI'] = df['RSI'] < 30
df['Sell_Signal_RSI'] = df['RSI'] > 70
# MACD 기반 매수/매도 신호
df['Buy_Signal_MACD'] = df['MACD'] > df['Signal_Line']
df['Sell_Signal_MACD'] = df['MACD'] < df['Signal_Line']


""" 3단계 시각화 """
# 시각화
plt.figure(figsize=(12, 8))

# 종가와 이동 평균
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.plot(df.index, df['SMA50'], label='50-day SMA', color='red')
plt.plot(df.index, df['SMA200'], label='200-day SMA', color='green')

# 매수/매도 신호 표시
plt.scatter(df.index[df['Buy_Signal_MA']], df['Close'][df['Buy_Signal_MA']], label='Buy Signal (MA)', marker='^', color='green')
plt.scatter(df.index[df['Sell_Signal_MA']], df['Close'][df['Sell_Signal_MA']], label='Sell Signal (MA)', marker='v', color='red')

plt.title('Bitcoin Price with Buy/Sell Signals (MA)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (KRW)', fontsize=12)
plt.legend()

# RSI 시각화
plt.subplot(3, 1, 2)
plt.plot(df.index, df['RSI'], label='RSI', color='purple')
plt.axhline(30, linestyle='--', color='green', label='Buy Signal (RSI 30)')
plt.axhline(70, linestyle='--', color='red', label='Sell Signal (RSI 70)')

plt.scatter(df.index[df['Buy_Signal_RSI']], df['RSI'][df['Buy_Signal_RSI']], label='Buy Signal (RSI)', marker='^', color='green')
plt.scatter(df.index[df['Sell_Signal_RSI']], df['RSI'][df['Sell_Signal_RSI']], label='Sell Signal (RSI)', marker='v', color='red')

plt.title('RSI with Buy/Sell Signals', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('RSI', fontsize=12)
plt.legend()

# MACD 시각화
plt.subplot(3, 1, 3)
plt.plot(df.index, df['MACD'], label='MACD', color='blue')
plt.plot(df.index, df['Signal_Line'], label='Signal Line', color='red')

plt.scatter(df.index[df['Buy_Signal_MACD']], df['MACD'][df['Buy_Signal_MACD']], label='Buy Signal (MACD)', marker='^', color='green')
plt.scatter(df.index[df['Sell_Signal_MACD']], df['MACD'][df['Sell_Signal_MACD']], label='Sell Signal (MACD)', marker='v', color='red')

plt.title('MACD with Buy/Sell Signals', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('MACD', fontsize=12)
plt.legend()

# 그래프 출력
plt.tight_layout()
plt.show()
