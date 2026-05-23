import pandas as pd
import numpy as np
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download resource NLTK (hanya perlu sekali)
nltk.download('stopwords', quiet=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================
def load_data():
    """Load dataset utama wisata Indonesia"""
    df = pd.read_csv('tourism_with_id.csv')
    print(f"✅ Dataset dimuat: {len(df)} baris")
    return df


# ============================================================
# 2. CLEANING DATA
# ============================================================
def clean_data(df):
    """Bersihkan kolom yang tidak diperlukan dan tangani missing value"""

    # Hapus kolom tidak relevan
    kolom_hapus = ['Unnamed: 11', 'Unnamed: 12', 'Coordinate']
    df = df.drop(columns=[k for k in kolom_hapus if k in df.columns])

    # Isi missing value Time_Minutes dengan median
    median_time = df['Time_Minutes'].median()
    df['Time_Minutes'] = df['Time_Minutes'].fillna(median_time)

    # Hapus duplikat jika ada
    df = df.drop_duplicates(subset='Place_Id')

    # Reset index
    df = df.reset_index(drop=True)

    print(f"✅ Data setelah cleaning: {len(df)} baris")
    print(f"   Missing values tersisa: {df.isnull().sum().sum()}")
    return df


# ============================================================
# 3. PREPROCESSING TEKS
# ============================================================

# Inisialisasi Sastrawi
factory_stem     = StemmerFactory()
stemmer          = factory_stem.create_stemmer()

factory_stop     = StopWordRemoverFactory()
stop_remover     = factory_stop.create_stop_word_remover()

def preprocess_teks(teks):
    """
    Preprocessing teks deskripsi wisata:
    1. Lowercase
    2. Hapus angka & karakter khusus
    3. Hapus stopword Bahasa Indonesia
    4. Stemming dengan PySastrawi
    """
    if pd.isna(teks):
        return ''

    # 1. Lowercase
    teks = teks.lower()

    # 2. Hapus angka dan karakter non-huruf
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)

    # 3. Hapus spasi berlebih
    teks = re.sub(r'\s+', ' ', teks).strip()

    # 4. Hapus stopword
    teks = stop_remover.remove(teks)

    # 5. Stemming
    teks = stemmer.stem(teks)

    return teks


def preprocess_dataset(df):
    """Terapkan preprocessing teks ke kolom Description"""
    print("⏳ Memproses teks deskripsi... (ini butuh 1-2 menit)")

    df['Description_Clean'] = df['Description'].apply(preprocess_teks)

    # Gabungkan Description + Category + City sebagai fitur utama
    df['Fitur_Gabungan'] = (
        df['Description_Clean'] + ' ' +
        df['Category'].str.lower() + ' ' +
        df['City'].str.lower()
    )

    print("✅ Preprocessing teks selesai!")
    return df


# ============================================================
# 4. TAMBAHKAN LABEL TIPE WISATA
# ============================================================
def tambah_tipe_wisata(df):
    """Tambahkan kolom tipe wisata berdasarkan keyword di nama tempat"""

    def deteksi_tipe(nama):
        nama = nama.lower()
        if any(k in nama for k in ['pantai', 'beach', 'laut', 'pulau']):
            return 'Pantai/Bahari'
        elif any(k in nama for k in ['gunung', 'bukit', 'puncak']):
            return 'Gunung/Bukit'
        elif any(k in nama for k in ['air terjun', 'curug', 'waterfall']):
            return 'Air Terjun'
        elif any(k in nama for k in ['goa', 'gua', 'cave']):
            return 'Goa'
        elif any(k in nama for k in ['danau', 'telaga', 'sungai', 'lake']):
            return 'Danau/Sungai'
        elif any(k in nama for k in ['alun']):
            return 'Alun-alun'
        elif any(k in nama for k in ['museum']):
            return 'Museum'
        elif any(k in nama for k in ['candi', 'keraton', 'istana', 'pura']):
            return 'Candi/Keraton'
        elif any(k in nama for k in ['kebun', 'taman', 'garden', 'zoo']):
            return 'Kebun/Taman'
        else:
            match = df[df['Place_Name'] == nama]['Category']
            return match.values[0] if len(match) > 0 else 'Lainnya'

    df['Tipe_Wisata'] = df['Place_Name'].apply(deteksi_tipe)
    print("✅ Label tipe wisata berhasil ditambahkan!")
    return df


# ============================================================
# 5. SIMPAN HASIL PREPROCESSING
# ============================================================
def simpan_data(df, nama_file='tourism_preprocessed.csv'):
    """Simpan dataset yang sudah dipreprocess"""
    df.to_csv(nama_file, index=False)
    print(f"✅ Data tersimpan sebagai '{nama_file}'")


# ============================================================
# MAIN - Jalankan semua langkah preprocessing
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("   PREPROCESSING DATASET WISATA INDONESIA")
    print("=" * 50)

    # Step 1: Load
    df = load_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Preprocessing teks
    df = preprocess_dataset(df)

    # Step 4: Tambah label tipe wisata
    df = tambah_tipe_wisata(df)

    # Step 5: Simpan
    simpan_data(df)

    print()
    print("=" * 50)
    print("✅ PREPROCESSING SELESAI!")
    print("=" * 50)
    print()
    print("Contoh hasil preprocessing:")
    print(df[['Place_Name', 'Category', 'Tipe_Wisata', 'Description_Clean']].head(3).to_string())
