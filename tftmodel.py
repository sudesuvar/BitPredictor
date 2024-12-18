import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Veriyi yükleme
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

# Verileri TensorFlow için uygun hale getirme
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # 3D Tensor
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))    # 3D Tensor

# Basit bir LSTM modelini kullanarak modelin temelini atıyoruz
model = tf.keras.Sequential([
    layers.LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.Dense(1)
])

# Modeli derleyelim
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitelim ve eğitim süresini ölçelim
start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
training_time = time.time() - start_time

# Çıkarım süresi ölçümü
start_time = time.time()
y_pred = model.predict(X_test)
inference_time = time.time() - start_time

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
print(f"Training Time: {training_time:.2f} seconds")
print(f"Inference Time: {inference_time:.2f} seconds")

# Gerçek fiyatlar ve tahmin edilen fiyatları karşılaştıran bir çizgi grafiği
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Gerçek Fiyat', color='blue', alpha=0.6)
plt.plot(y_pred, label='Tahmin Edilen Fiyat', color='red', linestyle='--', alpha=0.6)
plt.title('Gerçek ve Tahmin Edilen Bitcoin Fiyatları')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.grid(True)
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

# Hata (Gerçek Fiyat - Tahmin Edilen Fiyat) dağılımını gösteren histogram
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='orange', edgecolor='black')
plt.title('Model Hatası Dağılımı')
plt.xlabel('Hata (Gerçek - Tahmin)')
plt.ylabel('Frekans')
plt.grid(True)
plt.show()

# ROC Eğrisi ve Karmaşıklık Matrisi (Bu bölümler zaman serisi regresyonu için uygun değil, çıkarıldı)
# ROC ve karmaşıklık matrisi sınıflandırma problemleri içindir, bu nedenle zaman serisi regresyonu için geçerli değil.
