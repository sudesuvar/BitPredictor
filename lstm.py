import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Veri yükleme
data = pd.read_csv("/Users/zeynep/Desktop/yazlab/BitPredictor/btcData.csv", nrows=6470)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Fiyatı kaydırarak önceki fiyat bilgisini ekleyelim
data['Previous_Price'] = data['Price'].shift(1)
data.dropna(inplace=True)

# Bağımsız değişken ve bağımlı değişken
X = data[['Previous_Price']].values
y = data['Price'].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Veriyi 3D tensöre çevirelim (LSTM için gerekli)
def prepare_data(X, y, n_steps):
    X_new, y_new = [], []
    for i in range(len(X) - n_steps):
        X_new.append(X[i:i+n_steps])
        y_new.append(y[i+n_steps])
    return np.array(X_new), np.array(y_new)

n_steps = 7  # 7 gün geçmiş bilgisi ile tahmin yapılacak
X_train, y_train = prepare_data(X_train, y_train, n_steps)
X_test, y_test = prepare_data(X_test, y_test, n_steps)

# LSTM için veriyi uygun formata getiriyoruz
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# LSTM Modeli
model = tf.keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    layers.Dense(1)
])

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Eğitim zamanını ölçme
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

# Eğitim ve test loss grafikleri
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Loss)', color='blue')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Val Loss)', color='red')
plt.title('Loss vs Epochs (Eğitim ve Doğrulama)', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Gerçek fiyatlar ve tahmin edilen fiyatları karşılaştırma
dates = data.index[-len(y_test):]  # Tarihler y_test boyutunda
plt.figure(figsize=(14, 7))
plt.plot(dates, y_test, label='Gerçek Fiyat', color='blue', alpha=0.6)
plt.plot(dates, y_pred, label='Tahmin Edilen Fiyat', color='red', linestyle='--', alpha=0.6)
plt.title(' LSTM Gerçek ve Tahmin Edilen Bitcoin Fiyatları', fontsize=16)
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Hata yüzdesi grafiği
errors = (y_test - y_pred.flatten()) / y_test * 100
plt.figure(figsize=(14, 7))
plt.plot(dates, errors, label='Hata Payı (%)', color='purple', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Gerçek Fiyatlara Göre Tahmin Hatası (%)', fontsize=16)
plt.xlabel('Tarih')
plt.ylabel('Hata Payı (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Performans metriklerini grafiğin altına ekleme
error_metrics_text = (
    f"MSE: {mse:.2f}\n"
    f"MAE: {mae:.2f}\n"
    f"RMSE: {rmse:.2f}\n"
    f"R-Squared: {r2:.2f}\n"
    f"MAPE: {mape:.2f}%"
)
plt.gcf().text(0.1, 0.02, error_metrics_text, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

plt.show()