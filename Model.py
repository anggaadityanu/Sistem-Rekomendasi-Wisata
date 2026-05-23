import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 1. LOAD DATA HASIL PREPROCESSING
# ============================================================
def load_data():
    df = pd.read_csv('tourism_preprocessed.csv')
    print(f"✅ Data dimuat: {len(df)} tempat wisata")
    return df


# ============================================================
# 2. MODEL UTAMA: TF-IDF + COSINE SIMILARITY
# ============================================================
def build_tfidf_model(df):
    """Bangun model TF-IDF + Cosine Similarity"""
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigram dan bigram
        min_df=1
    )
    tfidf_matrix = tfidf.fit_transform(df['Fitur_Gabungan'].fillna(''))
    cosine_sim    = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"✅ Model TF-IDF selesai dibangun")
    print(f"   Ukuran matriks TF-IDF: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix, cosine_sim


# ============================================================
# 3. MODEL BASELINE 1: COUNT VECTORIZER + COSINE SIMILARITY
# ============================================================
def build_count_model(df):
    """Bangun model Count Vectorizer + Cosine Similarity (baseline)"""
    count_vec    = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    count_matrix = count_vec.fit_transform(df['Fitur_Gabungan'].fillna(''))
    cosine_sim   = cosine_similarity(count_matrix, count_matrix)
    print(f"✅ Model Count Vectorizer selesai dibangun")
    return count_vec, count_matrix, cosine_sim


# ============================================================
# 4. MODEL BASELINE 2: KNN
# ============================================================
def build_knn_model(tfidf_matrix, n_neighbors=11):
    """Bangun model KNN menggunakan matriks TF-IDF"""
    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric='cosine',
        algorithm='brute'
    )
    knn.fit(tfidf_matrix)
    print(f"✅ Model KNN selesai dibangun (k={n_neighbors})")
    return knn


# ============================================================
# 5. FUNGSI REKOMENDASI
# ============================================================
def rekomendasikan_tfidf(nama_tempat, df, cosine_sim, top_k=10):
    """
    Rekomendasikan tempat wisata menggunakan TF-IDF + Cosine Similarity
    """
    # Cari index tempat yang dicari
    nama_tempat = nama_tempat.strip()
    matches = df[df['Place_Name'].str.lower() == nama_tempat.lower()]

    if matches.empty:
        print(f"❌ Tempat '{nama_tempat}' tidak ditemukan!")
        return pd.DataFrame()

    idx = matches.index[0]

    # Hitung similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ambil top_k (skip index 0 karena itu dirinya sendiri)
    sim_scores = sim_scores[1:top_k+1]
    place_indices = [i[0] for i in sim_scores]
    scores        = [i[1] for i in sim_scores]

    # Buat dataframe hasil
    hasil = df.iloc[place_indices][['Place_Name', 'Category', 'Tipe_Wisata',
                                     'City', 'Rating', 'Price']].copy()
    hasil['Similarity_Score'] = scores
    hasil['Similarity_Score'] = hasil['Similarity_Score'].round(4)
    hasil = hasil.reset_index(drop=True)
    hasil.index += 1  # mulai dari 1

    return hasil


def rekomendasikan_count(nama_tempat, df, cosine_sim_count, top_k=10):
    """Rekomendasikan menggunakan Count Vectorizer + Cosine Similarity"""
    matches = df[df['Place_Name'].str.lower() == nama_tempat.lower()]
    if matches.empty:
        return pd.DataFrame()

    idx        = matches.index[0]
    sim_scores = list(enumerate(cosine_sim_count[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]

    place_indices = [i[0] for i in sim_scores]
    scores        = [i[1] for i in sim_scores]

    hasil = df.iloc[place_indices][['Place_Name', 'Category', 'City', 'Rating']].copy()
    hasil['Similarity_Score'] = [round(s, 4) for s in scores]
    hasil = hasil.reset_index(drop=True)
    hasil.index += 1
    return hasil


def rekomendasikan_knn(nama_tempat, df, knn_model, tfidf_matrix, top_k=10):
    """Rekomendasikan menggunakan KNN"""
    matches = df[df['Place_Name'].str.lower() == nama_tempat.lower()]
    if matches.empty:
        return pd.DataFrame()

    idx          = matches.index[0]
    distances, indices = knn_model.kneighbors(
        tfidf_matrix[idx], n_neighbors=top_k+1
    )

    # Skip index pertama (dirinya sendiri)
    indices   = indices[0][1:]
    distances = distances[0][1:]
    scores    = [round(1 - d, 4) for d in distances]  # ubah distance ke similarity

    hasil = df.iloc[indices][['Place_Name', 'Category', 'City', 'Rating']].copy()
    hasil['Similarity_Score'] = scores
    hasil = hasil.reset_index(drop=True)
    hasil.index += 1
    return hasil


# ============================================================
# 6. FUNGSI REKOMENDASI BERDASARKAN KATEGORI/TIPE
# ============================================================
def rekomendasikan_by_kategori(kategori, df, cosine_sim, top_k=10):
    """
    Rekomendasikan tempat wisata berdasarkan kategori atau tipe wisata.
    Misal: 'Pantai/Bahari', 'Gunung/Bukit', 'Budaya', dll
    """
    # Cari semua tempat dengan kategori tsb
    mask = (
        df['Category'].str.lower().str.contains(kategori.lower(), na=False) |
        df['Tipe_Wisata'].str.lower().str.contains(kategori.lower(), na=False)
    )
    matches = df[mask]

    if matches.empty:
        print(f"❌ Kategori '{kategori}' tidak ditemukan!")
        return pd.DataFrame()

    # Ambil rata-rata similarity dari semua tempat di kategori tsb
    idx_list   = matches.index.tolist()
    avg_scores = cosine_sim[idx_list].mean(axis=0)

    sim_scores = list(enumerate(avg_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Filter hanya yang termasuk kategori tsb
    hasil_idx    = [i for i, s in sim_scores if i in idx_list][:top_k]
    hasil_scores = [avg_scores[i] for i in hasil_idx]

    hasil = df.iloc[hasil_idx][['Place_Name', 'Category', 'Tipe_Wisata',
                                  'City', 'Rating', 'Price']].copy()
    hasil['Relevance_Score'] = [round(s, 4) for s in hasil_scores]
    hasil = hasil.sort_values('Rating', ascending=False).reset_index(drop=True)
    hasil.index += 1
    return hasil


# ============================================================
# MAIN - Test semua model
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("   MEMBANGUN MODEL REKOMENDASI WISATA")
    print("=" * 55)

    # Load data
    df = load_data()

    # Build semua model
    tfidf, tfidf_matrix, cosine_sim_tfidf   = build_tfidf_model(df)
    count, count_matrix, cosine_sim_count   = build_count_model(df)
    knn                                      = build_knn_model(tfidf_matrix)

    print()
    print("=" * 55)
    print("CONTOH REKOMENDASI - TF-IDF + Cosine Similarity")
    print("=" * 55)
    contoh = 'Monumen Nasional'
    print(f"\nTempat: {contoh}")
    print("-" * 55)
    hasil_tfidf = rekomendasikan_tfidf(contoh, df, cosine_sim_tfidf)
    print(hasil_tfidf[['Place_Name', 'Category', 'City',
                        'Rating', 'Similarity_Score']].to_string())

    print()
    print("=" * 55)
    print("CONTOH REKOMENDASI - Count Vectorizer + Cosine Similarity")
    print("=" * 55)
    hasil_count = rekomendasikan_count(contoh, df, cosine_sim_count)
    print(hasil_count.to_string())

    print()
    print("=" * 55)
    print("CONTOH REKOMENDASI - KNN")
    print("=" * 55)
    hasil_knn = rekomendasikan_knn(contoh, df, knn, tfidf_matrix)
    print(hasil_knn.to_string())

    print()
    print("=" * 55)
    print("CONTOH REKOMENDASI BERDASARKAN KATEGORI")
    print("=" * 55)
    hasil_kat = rekomendasikan_by_kategori('Pantai', df, cosine_sim_tfidf)
    print(hasil_kat.to_string())

    print()
    print("✅ Semua model berhasil dibangun dan diuji!")
    print("   Lanjut ke tahap EVALUASI -> python evaluasi.py")