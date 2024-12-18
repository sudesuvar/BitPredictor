import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Veri Yükleme ve Hazırlık
data = pd.read_csv("/Users/zeynep/Desktop/yazlab/BitPredictor/btcData.csv", nrows=6470)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Price'].plot(figsize=(14,7))
plt.title('Bitcoin Fiyatları Zaman Serisi')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.show()

# 2. İstasyonerlik Kontrolü ve Farklaştırma
result = adfuller(data['Price'])
print(f'ADF Testi İstatistiği: {result[0]}, p-değeri: {result[1]}')
if result[1] >= 0.05:
    data['Price_diff'] = data['Price'] - data['Price'].shift(1)
    data.dropna(inplace=True)

# 3. ARIMA Modelini Uygulama
model = ARIMA(data['Price'], order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())

# 4. Tahmin Sonuçları
forecast = model_fit.forecast(steps=10)
print("Tahmin Edilen Değerler:", forecast)

# 5. Performans Değerlendirme
mse = mean_squared_error(data['Price'][-10:], forecast)
mae = mean_absolute_error(data['Price'][-10:], forecast)
rmse = np.sqrt(mse)
r2 = r2_score(data['Price'][-10:], forecast)
print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2: {r2}")

# 6. Tahmin Sonuçlarını Görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Price'], label='Gerçek Fiyat', color='blue', alpha=0.6)
plt.plot(pd.date_range(data.index[-1], periods=11, freq='D')[1:], forecast, label='Tahmin Edilen Fiyat', color='red', linestyle='--', alpha=0.6)
plt.legend()
plt.show()
