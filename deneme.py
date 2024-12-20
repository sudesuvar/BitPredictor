import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# CSV dosyasını yükleme
#data = pd.read_csv("btcData.csv", nrows=6470)  # CSV dosyanızın adını buraya yazın
#data = pd.read_csv("/Users/zeynep/Desktop/yazlab/BitPredictor/btcData.csv", nrows=6470)
data = pd.read_csv("btcData.csv", nrows=6470)
print(data.head())

# Tarih ve fiyat sütunlarını kontrol et
# Örnek olarak, 'Date' ve 'Price' sütunları olduğunu varsayıyoruz
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Bağımsız değişken (X) ve bağımlı değişken (y) seçimi
data['Previous_Price'] = data['Price'].shift(1)  # Bir önceki günü ekliyoruz
data.dropna(inplace=True)

X = data[['Previous_Price']]  # Feature olarak bir önceki fiyat
y = data['Price']  # Tahmin edilecek fiyat

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Lineer regresyon modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Yüzde olarak
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Sonuçları yazdırma
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared: {r2}")

# Gerçek fiyatlar ve tahmin edilen fiyatları karşılaştıran bir çizgi grafiği
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Gerçek Fiyat', color='blue', alpha=0.6)
plt.plot(y_test.index, y_pred, label='Tahmin Edilen Fiyat', color='red', linestyle='--', alpha=0.6)
plt.title('Gerçek ve Tahmin Edilen Bitcoin Fiyatları')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.show()

# Hata (Gerçek Fiyat - Tahmin Edilen Fiyat) dağılımını gösteren histogram
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='orange', edgecolor='black')
plt.title('Model Hatası Dağılımı')
plt.xlabel('Hata (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.grid(True)
plt.show()
