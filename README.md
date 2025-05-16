# 🩺 Diabetes Prediction Project

This project aims to predict whether an individual has diabetes using machine learning techniques. A web-based interface collects user input, feeds it into a trained model, and displays the prediction result on the screen.


## 📁 Proje Yapısı
```
diabetes_project/
│
├── app.py # Flask ana uygulama dosyası
├── main.py # Alternatif başlatıcı dosya (varsa)
├── requirements.txt # Gerekli kütüphaneler
├── README.md # Bu dosya
│
├── data/
│ └── diabetes.csv # Veri kümesi
│
├── models/
│ ├── encoder.pkl
│ ├── scaler.pkl
│ ├── voting_clf.pkl
│ └── outlier_limits.pkl # Gerekli model bileşenleri
│
├── src/
│ ├── data_preprocessing.py
│ ├── new_data_preprocess.py
│ ├── model_prediction.py
│ ├── model_training.py
│ └── utils.py # Yardımcı fonksiyonlar
│
└── templates/
├── index.html # Ana giriş formu
└── predict.html # Sonuç sayfası
```

yaml
Kodu kopyala

---

## 🚀 Kurulum ve Çalıştırma

### 1. Sanal Ortam Oluştur
```
bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
```
2. Gerekli Kütüphaneleri Kur
```
bash
Kodu kopyala
pip install -r requirements.txt
```
3. Flask Uygulamasını Başlat
```
bash
Kodu kopyala
python app.py
```
4. Web Arayüzüne Git  
Tarayıcıdan şu adrese git:

```
http://127.0.0.1:5000
```
🧠 Model Özeti
Veri ön işleme:

Eksik değer doldurma

Aykırı değer temizleme

Kategorik değişken encode etme

Normalizasyon

Kullanılan model:

VotingClassifier (Lojistik Regresyon, Random Forest, XGBoost birleşimi)

Model bileşenleri models/ klasöründe pickle dosyaları olarak saklanır.

🛠️ Kullanılan Teknolojiler

| Teknoloji         | Açıklama                                                 |
| ----------------- | -------------------------------------------------------- |
| **Python 3.x**    | Projenin ana programlama dili                            |
| **Flask**         | Web arayüzünü oluşturmak için kullanılan mikro framework |
| **Pandas**        | Veri işleme ve analiz işlemleri için kullanıldı          |
| **Scikit-learn**  | Makine öğrenmesi algoritmaları ve modelleme için         |
| **XGBoost**       | Gelişmiş sınıflandırma modeli için kullanıldı            |
| **Pickle**        | Model ve ön işleme nesnelerini kaydetmek/yüklemek için   |
| **HTML & Jinja2** | Web sayfaları ve şablon motoru                           |

