import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import time

# CSV dosyasını yükleme
data = pd.read_csv("/Users/zeynep/Desktop/yazlab/BitPredictor/btcData.csv", nrows=6470)
print(data.head())

# Tarih ve fiyat sütunlarını kontrol et
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Bağımsız değişken (X) ve bağımlı değişken (y) seçimi
data['Previous_Price'] = data['Price'].shift(1)  # Bir önceki günü ekliyoruz
data.dropna(inplace=True)

X = data[['Previous_Price']]  # Feature olarak bir önceki fiyat
y = data['Price']  # Tahmin edilecek fiyat

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Model oluşturma ve eğitme
t_start_train = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
t_end_train = time.time()

# Tahminler
t_start_pred = time.time()
y_pred = model.predict(X_test)
t_end_pred = time.time()

# Performans metriklerini hesaplama
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Yüzde olarak
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Eğitim ve tahmin sürelerini ölçme
train_time = t_end_train - t_start_train
prediction_time = t_end_pred - t_start_pred

# 1. Grafik: Gerçek ve Tahmin Edilen Bitcoin Fiyatları
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Gerçek Fiyat', color='blue', alpha=0.6)
plt.plot(y_test.index, y_pred, label='Tahmin Edilen Fiyat', color='red', linestyle='--', alpha=0.6)
plt.title('Gerçek ve Tahmin Edilen Bitcoin Fiyatları', fontsize=14)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Fiyat', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.text(0.5, 1.1, f"MSE: {mse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | RMSE: {rmse:.2f} | R^2: {r2:.2f}\nEğitim Süresi: {train_time:.4f} s | Tahmin Süresi: {prediction_time:.4f} s",
         transform=plt.gca().transAxes, fontsize=10, ha='center', va='center', wrap=True)
plt.show()

# 2. Grafik: Hata Dağılımı
plt.figure(figsize=(14, 6))
errors = y_test - y_pred
plt.hist(errors, bins=50, color='orange', edgecolor='black')
plt.title('Model Hatası Dağılımı', fontsize=14)
plt.xlabel('Hata (Gerçek - Tahmin)', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.grid(True)
plt.show()

# 3. Grafik: Loss (MSE)
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_pred)), (y_test - y_pred)**2, color='purple', label='Loss (MSE)', alpha=0.6)
plt.title('Loss (MSE) Grafiği', fontsize=14)
plt.xlabel('Örnek', fontsize=12)
plt.ylabel('MSE Değeri', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# 4. Grafik: Hata Oranı Yüzdesel Dağılım
plt.figure(figsize=(14, 6))
error_percentage = (errors / y_test) * 100
plt.plot(y_test.index, error_percentage, color='green', label='Hata Yüzdesi', alpha=0.6)
plt.title('Hata Oranı Yüzdesi Grafiği', fontsize=14)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Hata Oranı (%)', fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()
