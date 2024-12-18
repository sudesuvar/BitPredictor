import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Veri yükleme
data = pd.read_csv("/Users/zeynep/Desktop/yazlab/BitPredictor/btcData.csv", nrows=6470)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Fiyatı kaydırarak önceki fiyat bilgisini ekleyelim
data['Previous_Price'] = data['Price'].shift(1)
data.dropna(inplace=True)

# Bağımsız değişken ve bağımlı değişken
X = data[['Previous_Price']].values  # Önceki fiyat
y = data['Price'].values  # Tahmin edilecek fiyat

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Veriyi 3D tensöre çevirelim (multi-step forecasting için reshape etme)
def prepare_data(X, y, n_steps):
    X_new, y_new = [], []
    for i in range(len(X) - n_steps):
        X_new.append(X[i:i+n_steps])
        y_new.append(y[i+n_steps])
    return np.array(X_new), np.array(y_new)

# 7 adımlı tahmin yapacağız (örneğin, 7 gün sonrasını tahmin etme)
n_steps = 7
X_train, y_train = prepare_data(X_train, y_train, n_steps)
X_test, y_test = prepare_data(X_test, y_test, n_steps)

# Veriyi 3D tensöre çevirelim
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))  # (n_samples, n_steps, n_features)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))  # (n_samples, n_steps, n_features)

# Autoformer modelini oluşturma (çok adımlı tahmin için)
class Autoformer(tf.keras.Model):
    def __init__(self, units=64):
        super(Autoformer, self).__init__()
        self.lstm = layers.LSTM(units, activation='relu', return_sequences=True)
        self.dense = layers.Dense(1)
        self.attention = layers.MultiHeadAttention(num_heads=2, key_dim=units)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.attention(x, x)
        return self.dense(x[:, -1, :])  # Çıkışı sadece son adım için alıyoruz

# Modeli derleyelim
model = Autoformer()

model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitelim
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Sonuçları yazdırma
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared: {r2}")

# Gerçek fiyatlar ve tahmin edilen fiyatları karşılaştıran bir çizgi grafiği
plt.figure(figsize=(14, 7))

# x ekseni olarak tarihler
dates = data.index[-len(y_test):]  # y_test boyutuyla uyumlu son tarihleri al

plt.plot(dates, y_test, label='Gerçek Fiyat', color='blue', alpha=0.6)
plt.plot(dates, y_pred, label='Tahmin Edilen Fiyat', color='red', linestyle='--', alpha=0.6)

plt.suptitle('Autoformer Modeli: Gerçek ve Tahmin Edilen Bitcoin Fiyatları', fontsize=16)
plt.title('Gerçek ve Tahmin Edilen Fiyatlar', fontsize=14)
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Tarihler daha okunabilir olması için döndürülebilir
plt.show()

# Eğitim süreci grafiklerini çizme
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
