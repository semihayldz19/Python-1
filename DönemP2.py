import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veri setini yükle
hava = pd.read_csv("machine2.csv", delimiter=";", encoding="latin-1")

# Tarih sütununu datetime formatına dönüştür
hava["TARIH"] = pd.to_datetime(hava["TARIH"], format="%d.%m.%Y")

# Tarih sütununu Unix zaman damgalarına dönüştür
hava["TARIH"] = (hava["TARIH"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

# Bağımlı ve bağımsız değişkenleri ayır
X = hava.drop(["TARIH"], axis=1)
Y = hava["TARIH"]

# Veri setini eğitim ve test olarak ayır
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Lineer regresyon modelini oluştur ve eğit
model = LinearRegression()
model.fit(X_train, Y_train)

# Eğitim verisi üzerinde tahmin yap ve hata hesapla
Y_train_pred = model.predict(X_train)
train_error = mean_squared_error(Y_train, Y_train_pred)
print("Eğitim Hatası:", train_error)

# Test verisi üzerinde tahmin yap ve hata hesapla
Y_test_pred = model.predict(X_test)
test_error = mean_squared_error(Y_test, Y_test_pred)
print("Test Hatası:", test_error)

# Gerçek ve tahmin değerlerinin dağılımını göster
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_test_pred, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linewidth=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Değerler')
plt.title('Gerçek ve Tahmin Değerlerinin Dağılımı')
plt.show()

# R^2 skoru hesapla
r2 = r2_score(Y_test, Y_test_pred)
print("R^2 Skoru:", r2)
