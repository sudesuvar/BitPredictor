import pandas as pd
import os

# 1. İlk veri kümesi: btcData.csv (Saatlik fiyatlar - Date ve Price)
df1 = pd.read_csv('btcData.csv', on_bad_lines='skip')
df1.columns = df1.columns.str.strip()  # Sütun adlarında boşlukları temizle

# Tarih formatını uygun hale getiriyoruz
df1['Date'] = pd.to_datetime(df1['Date']).dt.date  # Yalnızca tarihi alıyoruz

# 'Price' sütunundaki virgülleri kaldırıp sayıya dönüştürüyoruz
df1['Price'] = df1['Price'].replace({',': ''}, regex=True).astype(float)

# 2. İkinci veri kümesi: btcData1.csv (Günlük fiyatlar - Date, Open, High, Low, Close)
df2 = pd.read_csv('btcData1.csv', on_bad_lines='skip')
df2.columns = df2.columns.str.strip()  # Sütun adlarında boşlukları temizle

# Tarih formatını uygun hale getiriyoruz
df2['Date'] = pd.to_datetime(df2['Date'], format='%b %d, %Y').dt.date  # Yalnızca tarihi alıyoruz

# Sayısal verilerdeki virgülleri kaldırıp sayıya dönüştürüyoruz
for col in ['Open', 'High', 'Low', 'Close']:
    df2[col] = df2[col].replace({',': ''}, regex=True).astype(float)

# Veriyi kontrol edelim
print("İkinci Veri Kümesi (Günlük Fiyatlar):")
print(df2.head())

# 3. İki veri kümesini birleştirelim (tüm tarihleri almak için 'outer' kullanıyoruz)
df_combined = pd.merge(df1, df2, on='Date', how='outer')

# Sonuçları yazdıralım
print("\nBirleştirilmiş Veri Kümesi:")
print(df_combined)

# Temizlenmiş veriyi yeni bir CSV dosyasına kaydedin
cleaned_file = 'cleanData.csv'
df_combined.to_csv(cleaned_file, index=False)

# cleanData dosyasının başarıyla oluşturulduğunu kontrol et
if os.path.exists(cleaned_file):
    # Dosyaları silme işlemi
    os.remove('btcData.csv')
    os.remove('btcData1.csv')
    print("\nbtcData.csv ve btcData1.csv dosyaları silindi.")
else:
    print(f"{cleaned_file} dosyası oluşturulamadı. Dosyalar silinmedi.")
