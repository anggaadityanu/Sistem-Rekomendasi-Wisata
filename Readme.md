# 🌿 WisataKu — Sistem Rekomendasi Wisata Indonesia

Sistem rekomendasi pariwisata cerdas untuk menemukan destinasi wisata Indonesia terbaik berdasarkan preferensi pengguna. Aplikasi ini menggunakan machine learning (TF-IDF, Cosine Similarity, dan KNN) untuk memberikan rekomendasi yang akurat dan relevan.

---

## 📋 Daftar Isi
- [Fitur Utama](#fitur-utama)
- [Prerequisites](#prerequisites)
- [Instalasi](#instalasi)
- [Cara Menjalankan Program](#cara-menjalankan-program)
- [Struktur File](#struktur-file)
- [Dataset](#dataset)

---

## ✨ Fitur Utama

✅ **Rekomendasi Wisata Pintar** - Dapatkan rekomendasi berdasarkan preferensi dan riwayat rating
✅ **Multiple Algoritma ML** - TF-IDF + Cosine Similarity, KNN, Collaborative Filtering
✅ **User-Friendly Interface** - Aplikasi web interaktif dengan Streamlit
✅ **Data Preprocessing** - Text cleaning, tokenization, stemming bahasa Indonesia
✅ **Evaluasi Model** - Metrik performa (precision, recall, RMSE, MAE)
✅ **Exploratory Data Analysis** - Notebook Jupyter untuk analisis mendalam

---

## 📦 Prerequisites

Pastikan sudah menginstal:
- **Python 3.8 atau lebih tinggi**
- **pip** (Package manager Python)

Cek versi Python:
```bash
python --version
```

---

## 🚀 Instalasi

### 1️⃣ Clone atau Unduh Repository
```bash
cd c:\perkuliahan\sistem-wisata
```

### 2️⃣ Buat Virtual Environment (Opsional tapi Direkomendasikan)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Atau install manual dengan perintah:
```bash
pip install streamlit pandas numpy scikit-learn nltk sastrawi openpyxl
```

**Penjelasan dependencies:**
| Package | Versi | Fungsi |
|---------|-------|--------|
| streamlit | >=1.28.0 | Framework untuk membuat web app |
| pandas | >=2.0.0 | Data manipulation dan analysis |
| numpy | >=1.24.0 | Numerical computing |
| scikit-learn | >=1.3.0 | Machine learning algorithms |
| nltk | >=3.8.0 | Natural Language Processing |
| sastrawi | >=1.0.1 | Stemming & stopword removal bahasa Indonesia |
| openpyxl | >=3.10.0 | Read/write Excel files |

### 4️⃣ Download NLTK Data (Hanya Sekali)

```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## ▶️ Cara Menjalankan Program

### 🎯 Jalankan Aplikasi Web (Main Program)

```bash
streamlit run App.py
```

Aplikasi akan otomatis membuka browser di `http://localhost:8501`

### 📊 Jalankan Preprocessing Data

Jika ingin memproses data dari awal:
```bash
python preprocessing.py
```
Output: `tourism_preprocessed.csv`

### 🤖 Jalankan Model Training

Untuk melatih model dan menyimpan hasilnya:
```bash
python Model.py
```

### 📈 Jalankan Evaluasi Model

Untuk melihat performa model (Precision, Recall, RMSE, MAE):
```bash
python Evaluasi.py
```

### 📓 Jalankan Jupyter Notebook

Untuk exploratory data analysis:
```bash
jupyter notebook EDA_Wisata_Indonesia.ipynb
```

---

## 📁 Struktur File

```
sistem-wisata/
│
├── App.py                          # 🎯 Main program - Aplikasi Streamlit
├── Model.py                        # 🤖 Model training & building
├── Evaluasi.py                     # 📊 Model evaluation
├── preprocessing.py                # 🔧 Data preprocessing
├── EDA_Wisata_Indonesia.ipynb      # 📓 Exploratory Data Analysis
│
├── Data Files:
├── tourism_with_id.csv             # Data wisata original
├── tourism_preprocessed.csv        # Data setelah preprocessing
├── tourism_rating.csv              # Data rating pengguna
├── user.csv                        # Data pengguna
├── package_tourism.csv             # Data paket wisata
├── hasil_evaluasi.csv              # Hasil evaluasi model
│
├── Readme.md                       # Dokumentasi ini
└── requirements.txt                # Dependency list
```

---

## 💾 Dataset

### File Dataset yang Digunakan:

1. **tourism_with_id.csv** - Data wisata lengkap dengan ID
   - Columns: Place_Id, Place_Name, Description, Category, Rating, Latitude, Longitude, dll

2. **tourism_preprocessed.csv** - Data setelah preprocessing
   - Columns: Cleaned text, descriptions, categories

3. **tourism_rating.csv** - Rating dari pengguna
   - Columns: User_Id, Place_Id, Rating, Review_Text

4. **user.csv** - Profil pengguna
   - Columns: User_Id, User_Name, Location, dll

---

## 🛠️ Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solusi:**
```bash
pip install streamlit
```

### ❌ Error: "ModuleNotFoundError: No module named 'sastrawi'"
**Solusi:**
```bash
pip install sastrawi
```

### ❌ Port 8501 sudah terpakai
**Solusi:**
```bash
streamlit run App.py --server.port 8502
```

### ❌ Virtual environment tidak activate
**Solusi Windows:**
```bash
venv\Scripts\activate
```

---

## 📚 Quick Start

**Langkah tercepat untuk menjalankan aplikasi:**

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run aplikasi
streamlit run App.py
```

Aplikasi siap digunakan! 🎉

---

## 🔗 Kontak & Info Lebih Lanjut

Untuk pertanyaan atau umpan balik, silakan hubungi pengembang.

---

**Dibuat dengan ❤️ untuk Indonesia**