📘 Judul Proyek
*Analisis dan Prediksi Hasil Pertandingan Tenis Turnamen Major Menggunakan Machine Learning dan Deep Learning*

## 👤 Informasi
* **Nama:** Ramizah Budi C.P
* **Repo:** https://github.com/ramizahhbudi/234311049_UAS_Data-Science 
* **Video:** https://youtu.be/BMJJdi24RR8 

---

# 1. 🎯 Ringkasan Proyek
*Memprediksi hasil pertandingan (Menang/Kalah) berdasarkan statistik teknis permainan tenis.
*Melakukan data preparation (cleaning missing values, feature selection, scaling, splitting).
*Membangun 3 pendekatan model: (1) Baseline – Logistic Regression (2) Advanced ML – Random Forest Classifier (3) Deep Learning – MLP Neural Network (Binary Classification)
*Melakukan evaluasi menggunakan Accuracy, Precision, Recall, dan F1-Score.
*Menentukan model terbaik dan memberikan insight bahwa statistik poin total dan error adalah kunci kemenangan.

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
1.	Bagaimana distribusi kemenangan antara Pemain 1 dan Pemain 2 dalam dataset turnamen major?
2.	Fitur statistik permainan apa (seperti Ace, Double Fault, Unforced Error) yang paling membedakan antara pemain yang menang dan kalah?
3.	Model machine learning mana (antara Logistic Regression, Random Forest, atau MLP) yang paling akurat dalam memprediksi kemenangan?

**Goals:**  
1.	Menganalisis pola data pertandingan tenis melalui Exploratory Data Analysis (EDA).
2.	Membangun model klasifikasi untuk memprediksi kolom Result (Kemenangan Pemain 1) dengan akurasi > 85%
3.	Membandingkan performa model Baseline, Advanced, dan Deep Learning.

---
## 📁 Struktur Folder
```
234311049_UAS_DataScience/
│
├── data/
│   ├── AusOpen-men-2013 
│   └── AusOpen-women-2013                   
│   └── FrenchOpen-men-2013
│   └── FrenchOpen-women-2013
│   └── USOpen-men-2013
│   └── USOpen-women-2013
│   └── Wimbledon-men-2013
│   └── Wimbledon-women-2013
│   
├── notebooks/       
│   └──234311049_Ramizah Budi C_P_UAS.ipynb
│
├── src/                   
│   
├── models/                 
│   ├── random_forest_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── mlp_model.h5
│
├── images/                 
│   ├── Perbandingan ketiga model.png
│   ├── Accuracy&Loss.png
│   ├── Confusion Matrix Model 3.png
│   ├── Confusion Matrix Model 2.png
│   ├── Confusion Matrix Model 1.png
│   ├── Feature Important.png
│   ├── Visualisasi 2 (EDA) Korelasi Heatmap.png
│   └── Visualisasi 1 (EDA) Distribusi Fitur.png
│   └── visualisasi 3 (EDA) Boxplot.png
├── requirements.txt
├── Checklist Submit.md
├── LICENSE        
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
* **Sumber:** UCI Machine Learning Repository (Tennis Major Tournament Match Statistics) 
* **Jumlah Data:**  943 baris, 43 fitur awal (direduksi menjadi 19 fitur input + 1 target).
* **Tipe:** tabular

### Fitur Utama
Dataset ini menggunakan 19 fitur statistik teknis yang paling relevan.  
Berikut tabel fitur yang digunakan:

| **Nama Fitur**        | **Deskripsi** |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **Result**   | Target – Hasil pertandingan (0: Kalah, 1: Menang).                              |
| **ACE.1 / ACE.2**      | Jumlah servis Ace (poin langsung dari servis tanpa tersentuh lawan).                                         |
| **DBF.1 / DBF.2**       | Jumlah Double Faults (kesalahan ganda saat servis).                                                      |
| **WNR.1 / WNR.2**  | Jumlah Winners (pukulan kemenangan yang tidak bisa dikembalikan lawan).                            |
| **UFE.1 / UFE.2**      | Jumlah Unforced Errors (kesalahan sendiri/bola mati bukan karena tekanan lawan).                                             |
| **BPC.1 / BPC.2**         | Break Points Created (peluang mematahkan servis lawan).                                                                |
| **BPW.1 / BPW.2**     | Break Points Won (jumlah break point yang berhasil dimenangkan).                                                                                    |
| **NPA.1 / NPA.2**         | Net Points Attempted (jumlah maju ke depan net).                                                                           |
| **NPW.1 / NPW.2**         | Net Points Won (poin yang dimenangkan di depan net).                                                                                                        |
| **TPW.1 / TPW.2**         | Total Points Won (total poin yang dimenangkan sepanjang pertandingan).                                                                                                   |

---

# 4. 🔧 Data Preparation
* **Cleaning**: Mengisi missing values (NaN) dengan nilai 0 (asumsi jika data tidak tercatat berarti kejadiannya 0).
* **Feature Selection**:Memilih 19 fitur statistik teknis dan membuang kolom metadata (Nama Pemain, Round, Turnamen)
* **Scaling**: Menerapkan StandardScaler untuk menormalisasi rentang data (sangat penting untuk MLP dan Logistic Regression).
* **Splitting**: Pembagian dataset menjadi 80% train dan 20% test menggunakan train_test_split dengan random_state = 42.
* **Balancing**: Data target (Result) sudah seimbang (balanced), tidak memerlukan teknik resampling.  

---

# 5. 🤖 Modeling
* **Model 1 – Baseline:** [Logistic Regression]  
* **Model 2 – Advanced ML:** [Random Forest Classifier]  
* **Model 3 – Deep Learning:** [MultiLayer Perceptron]  

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy, Precision, Recall, F1-Score

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Logistic Regression)** | 0.89 | 0.89 | 0.89 | 0.89 | < 1 detik |
| **Advanced (Random Forest)** | 0.94 | 0.94 | 0.94 | 0.94 | ~2 detik |
| **Deep Learning (MLP)** | 0.88 | 0.87 | 0.89 | 0.88 | ~45 detik |

---

# 7. 🏁 Kesimpulan

* **Model terbaik:** Random Forest (Advanced ML)  
* **Alasan:**  
  - Mencapai akurasi tertinggi (94%) dibandingkan model lainnya.
  - Sangat efektif menangani hubungan non-linear antar fitur statistik tenis.
  - Lebih stabil dan cepat dilatih dibandingkan Deep Learning pada dataset ukuran ini (< 1000 baris). 

* **Insight penting:**  
  - Total Points Won (TPW) adalah indikator mutlak kemenangan.
  - Unforced Errors (UFE) memiliki dampak negatif besar; meminimalkan kesalahan sendiri lebih penting daripada mencetak banyak Winners.
  - Deep Learning (MLP) performanya sedikit di bawah Random Forest, menunjukkan bahwa untuk data tabular kecil, algoritma Ensemble Tree seringkali lebih unggul daripada Neural Networks.


---

# 8. 🔮 Future Work

## 📌 Data Improvements
* [x] Mengumpulkan lebih banyak data  
* [x] Menambah variasi data  
* [x] Melakukan feature engineering lebih lanjut  

## 🤖 Model Enhancements
* [x] Mencoba arsitektur deep learning yang lebih kompleks  
* [x] Hyperparameter tuning lebih ekstensif  
* [x] Mencoba ensemble methods  
* [ ] Transfer learning dengan model yang lebih besar  

## 🚀 Deployment & System
* [ ] Membuat API (Flask / FastAPI)  
* [x] Membuat web app (Streamlit / Gradio)  
* [ ] Containerization dengan Docker  
* [ ] Deploy ke cloud (Heroku / GCP / AWS)  

## ⚙️ Optimization
* [ ] Model compression (pruning / quantization)  
* [ ] Improving inference speed  
* [ ] Reducing model size  

---

# 9. 🔁 Reproducibility
Gunakan environment:
Python Version:
* Python 3.10

Main Libraries & Versions:
* numpy==1.24.3
* pandas==2.0.3
* scikit-learn==1.3.0
* matplotlib==3.7.1
* seaborn==0.12.2

Deep Learning Framework:
* tensorflow==2.15.0
* keras (termaasuk di tensorflow)
#
