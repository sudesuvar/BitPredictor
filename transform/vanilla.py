import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

# CSV dosyasını yükleyin
#data = pd.read_csv("/Users/zeynep/Desktop/yazlab/BitPredictor/btcData.csv", nrows=6470)
data = pd.read_csv("cleanData.csv")

# Veriyi hazırlama
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Fiyatı kaydırarak önceki fiyat bilgisini ekleyelim
data['Previous_Price'] = data['Price'].shift(1)
data.dropna(inplace=True)

# Veriyi normalleştirme
scaler = MinMaxScaler()
data[['Previous_Price', 'Price']] = scaler.fit_transform(data[['Previous_Price', 'Price']])

# Bağımsız değişken ve bağımlı değişken
X = data[['Previous_Price']].values
y = data['Price'].values

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Veriyi 3D tensöre çevirelim
def prepare_data(X, y, n_steps):
    X_new, y_new = [], []
    for i in range(len(X) - n_steps):
        X_new.append(X[i:i + n_steps])
        y_new.append(y[i + n_steps])
    return np.array(X_new), np.array(y_new)

n_steps = 7
X_train, y_train = prepare_data(X_train, y_train, n_steps)
X_test, y_test = prepare_data(X_test, y_test, n_steps)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Vanilla Transformer Modeli
class VanillaTransformer(tf.keras.Model):
    def __init__(self, seq_length, input_dim, units=64, num_heads=4, ff_dim=128, dropout_rate=0.1):
        super(VanillaTransformer, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.units = units
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Pozisyonel Kodlama
        self.positional_encoding = self._positional_encoding(seq_length, input_dim)

        # Embedding Layer
        self.embedding = layers.Dense(units)

        # Transformer Encoder Layer
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm1 = layers.LayerNormalization()

        # Feed-Forward Network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(units)
        ])
        self.dropout2 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization()

        # Output Layer
        self.final_dense = layers.Dense(1)

    def _positional_encoding(self, seq_length, input_dim):
        positions = np.arange(seq_length)[:, np.newaxis]
        dimensions = np.arange(input_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(input_dim))
        angle_rads = positions * angle_rates

        # Apply sin to even indices and cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        return tf.cast(angle_rads, dtype=tf.float32)

    def call(self, inputs):
        # Add positional encoding
        x = inputs + self.positional_encoding

        # Embedding
        x = self.embedding(x)

        # Self-Attention
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)  # Residual connection

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)  # Residual connection

        # Output Layer (Son zaman dilimi için tahmin)
        return self.final_dense(out2[:, -1, :])

# Modeli oluştur
model = VanillaTransformer(seq_length=n_steps, input_dim=X_train.shape[2])
model.compile(optimizer='adam', loss='mean_squared_error')

# Eğitim
start_training = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
training_time = time.time() - start_training

# Tahmin ve değerlendirme
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

# Sonuçları yazdırma
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}%")
print(f"Eğitim Süresi: {training_time:.2f} saniye")

# Gerçek ve tahmin edilen fiyatlar grafiği
plt.figure(figsize=(12, 8))
plt.plot(data.index[-len(y_test):], y_test, label='Gerçek Fiyatlar')
plt.plot(data.index[-len(y_pred):], y_pred, label='Tahmin Edilen Fiyatlar')
plt.title('Gerçek ve Tahmin Edilen Fiyatlar')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.legend()
plt.show()

# Eğitim ve doğrulama kaybı grafiği
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Metrikleri ve zamanları yazdıran grafik
fig, ax = plt.subplots(figsize=(14, 8))

metrics = {
    'MSE': mse,
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'MAPE (%)': mape,
    'Eğitim Süresi (s)': training_time
}

ax.table(cellText=list(metrics.items()), colLabels=['Metric', 'Value'], loc='center', cellLoc='center', colColours=['lightgray', 'lightgray'])
ax.axis('off')  # Disable axis

plt.title('Model Metrikleri ve Zaman')
plt.show()