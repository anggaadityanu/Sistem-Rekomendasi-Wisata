import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    df = pd.read_csv('tourism_preprocessed.csv')
    rating = pd.read_csv('tourism_rating.csv')
    print(f"✅ Data wisata  : {len(df)} tempat")
    print(f"✅ Data rating  : {len(rating)} rating")
    return df, rating


# ============================================================
# 2. BUILD MODELS
# ============================================================
def build_models(df):
    # TF-IDF + Cosine Similarity
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['Fitur_Gabungan'].fillna(''))
    cosine_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Count Vectorizer + Cosine Similarity
    count = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    count_matrix = count.fit_transform(df['Fitur_Gabungan'].fillna(''))
    cosine_count = cosine_similarity(count_matrix, count_matrix)

    # KNN
    knn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)

    print("✅ Semua model berhasil dibangun")
    return tfidf_matrix, cosine_tfidf, cosine_count, knn


# ============================================================
# 3. BUAT GROUND TRUTH DARI RATING
# ============================================================
def buat_ground_truth(df, rating, min_rating=3, min_reviews=3):
    """
    Ground truth: tempat wisata dianggap relevan jika memiliki
    rating >= min_rating dari user yang sama yang menilai tempat referensi.
    Relevansi berdasarkan kesamaan kategori + rating tinggi.
    """
    # Gabungkan rating dengan info tempat
    merged = rating.merge(
        df[['Place_Id', 'Category', 'City']],
        on='Place_Id', how='left'
    )

    # Ground truth berbasis kategori: tempat relevan = kategori sama + rating tinggi
    ground_truth = {}
    for idx, row in df.iterrows():
        kategori = row['Category']
        place_id = row['Place_Id']

        # Tempat relevan: kategori sama, rating >= min_rating, bukan dirinya sendiri
        relevan_ids = merged[
            (merged['Category'] == kategori) &
            (merged['Place_Ratings'] >= min_rating) &
            (merged['Place_Id'] != place_id)
        ]['Place_Id'].unique().tolist()

        if len(relevan_ids) >= min_reviews:
            ground_truth[idx] = relevan_ids

    print(f"✅ Ground truth dibuat: {len(ground_truth)} query tersedia")
    return ground_truth


# ============================================================
# 4. FUNGSI METRIK EVALUASI
# ============================================================
def precision_at_k(recommended_ids, relevant_ids, k):
    """Precision@K: seberapa banyak rekomendasi yang relevan"""
    recommended_k = recommended_ids[:k]
    hits = len(set(recommended_k) & set(relevant_ids))
    return hits / k if k > 0 else 0


def recall_at_k(recommended_ids, relevant_ids, k):
    """Recall@K: seberapa banyak yang relevan berhasil ditemukan"""
    recommended_k = recommended_ids[:k]
    hits = len(set(recommended_k) & set(relevant_ids))
    return hits / len(relevant_ids) if len(relevant_ids) > 0 else 0


def f1_at_k(precision, recall):
    """F1-Score@K: harmonic mean precision dan recall"""
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def average_precision(recommended_ids, relevant_ids, k):
    """Average Precision untuk satu query"""
    hits = 0
    sum_precision = 0
    relevant_set = set(relevant_ids)

    for i, rec_id in enumerate(recommended_ids[:k]):
        if rec_id in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / min(len(relevant_ids), k) if len(relevant_ids) > 0 else 0


# ============================================================
# 5. FUNGSI REKOMENDASI (untuk evaluasi)
# ============================================================
def get_recommendations_tfidf(idx, df, cosine_sim, top_k=10):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices]['Place_Id'].tolist()


def get_recommendations_count(idx, df, cosine_count, top_k=10):
    sim_scores = list(enumerate(cosine_count[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices]['Place_Id'].tolist()


def get_recommendations_knn(idx, df, knn, tfidf_matrix, top_k=10):
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=top_k+1)
    indices = indices[0][1:]
    return df.iloc[indices]['Place_Id'].tolist()


# ============================================================
# 6. EVALUASI SEMUA MODEL
# ============================================================
def evaluasi_model(nama_model, get_rec_func, df, ground_truth, k=10):
    """Evaluasi satu model dengan semua metrik"""
    precisions, recalls, f1s, aps = [], [], [], []

    for idx, relevant_ids in ground_truth.items():
        recommended_ids = get_rec_func(idx)

        p = precision_at_k(recommended_ids, relevant_ids, k)
        r = recall_at_k(recommended_ids, relevant_ids, k)
        f = f1_at_k(p, r)
        ap = average_precision(recommended_ids, relevant_ids, k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        aps.append(ap)

    hasil = {
        'Model'       : nama_model,
        f'Precision@{k}': round(np.mean(precisions), 4),
        f'Recall@{k}'   : round(np.mean(recalls), 4),
        f'F1-Score@{k}' : round(np.mean(f1s), 4),
        'MAP'           : round(np.mean(aps), 4),
    }
    return hasil


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("   EVALUASI MODEL REKOMENDASI WISATA")
    print("=" * 55)

    K = 10  # nilai K untuk evaluasi

    # Load data
    df, rating = load_data()

    # Build models
    tfidf_matrix, cosine_tfidf, cosine_count, knn = build_models(df)

    # Buat ground truth
    print("\n⏳ Membuat ground truth...")
    ground_truth = buat_ground_truth(df, rating)

    # Evaluasi masing-masing model
    print(f"\n⏳ Mengevaluasi semua model pada K={K}...")

    hasil_tfidf = evaluasi_model(
        f'TF-IDF + Cosine Similarity',
        lambda idx: get_recommendations_tfidf(idx, df, cosine_tfidf, K),
        df, ground_truth, K
    )

    hasil_count = evaluasi_model(
        f'Count Vectorizer + Cosine Similarity',
        lambda idx: get_recommendations_count(idx, df, cosine_count, K),
        df, ground_truth, K
    )

    hasil_knn = evaluasi_model(
        f'KNN (k=10)',
        lambda idx: get_recommendations_knn(idx, df, knn, tfidf_matrix, K),
        df, ground_truth, K
    )

    # Tampilkan hasil
    print("\n")
    print("=" * 65)
    print(f"   HASIL EVALUASI (K={K})")
    print("=" * 65)

    hasil_df = pd.DataFrame([hasil_tfidf, hasil_count, hasil_knn])
    hasil_df = hasil_df.set_index('Model')
    print(hasil_df.to_string())

    print("\n" + "=" * 65)

    # Tentukan model terbaik
    best_map   = hasil_df['MAP'].idxmax()
    best_f1    = hasil_df[f'F1-Score@{K}'].idxmax()
    best_prec  = hasil_df[f'Precision@{K}'].idxmax()

    print(f"\n🏆 Model terbaik berdasarkan:")
    print(f"   Precision@{K} : {best_prec}")
    print(f"   F1-Score@{K}  : {best_f1}")
    print(f"   MAP           : {best_map}")

    # Simpan hasil evaluasi
    hasil_df.to_csv('hasil_evaluasi.csv')
    print(f"\n✅ Hasil evaluasi tersimpan di 'hasil_evaluasi.csv'")
    print(f"✅ EVALUASI SELESAI! Lanjut ke -> python app.py (Streamlit)")