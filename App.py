import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="WisataKu — Rekomendasi Wisata Indonesia",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #F7F9F4; color: #1C2B1A; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1300px; }

/* Sidebar toggle — make it visible */
[data-testid="collapsedControl"] {
    background: #2D6A4F !important;
    border-radius: 0 8px 8px 0 !important;
    top: 50% !important;
    width: 28px !important;
    height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="collapsedControl"] svg { fill: white !important; width: 16px !important; }

.hero {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 45%, #52B788 100%);
    border-radius: 24px; padding: 3rem 3rem; margin: 1.5rem 0 2rem 0;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:300px; height:300px; background:rgba(255,255,255,0.05); border-radius:50%;
}
.hero-tag {
    display:inline-block; background:rgba(255,255,255,0.15); color:#B7E4C7;
    font-size:0.75rem; font-weight:600; letter-spacing:2px; text-transform:uppercase;
    padding:5px 14px; border-radius:20px; margin-bottom:1rem;
}
.hero h1 { font-family:'DM Serif Display',serif; font-size:2.8rem; color:#fff; line-height:1.15; margin-bottom:0.8rem; }
.hero h1 em { color:#95D5B2; font-style:italic; }
.hero p { font-size:1rem; color:rgba(255,255,255,0.75); max-width:480px; line-height:1.6; }
.hero-stats { display:flex; gap:2.5rem; margin-top:2rem; }
.hero-stat-val { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#fff; line-height:1; }
.hero-stat-lbl { font-size:0.78rem; color:rgba(255,255,255,0.6); margin-top:2px; }

.stTabs [data-baseweb="tab-list"] { background:transparent; gap:0.5rem; border-bottom:2px solid #D8E8D0; }
.stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif; font-weight:500; font-size:0.9rem; color:#6B8C6A; background:transparent; border-radius:8px 8px 0 0; padding:0.6rem 1.4rem; border:none; }
.stTabs [aria-selected="true"] { color:#1B4332 !important; background:#E8F5E0 !important; border-bottom:3px solid #2D6A4F !important; font-weight:600 !important; }

.place-card {
    background:#fff; border-radius:16px; overflow:hidden; margin-bottom:0.5rem;
    box-shadow:0 2px 12px rgba(27,67,50,0.07); display:flex;
    transition: transform 0.2s, box-shadow 0.2s;
}
.place-card:hover { transform:translateY(-2px); box-shadow:0 8px 28px rgba(27,67,50,0.13); }
.card-img { width:120px; min-height:140px; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:3rem; background:linear-gradient(135deg,#D8F3DC,#B7E4C7); }
.card-body { padding:1rem 1.2rem; flex:1; }
.card-rank { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#C8E6C0; float:right; line-height:1; }
.card-name { font-family:'DM Serif Display',serif; font-size:1.15rem; color:#1C2B1A; margin-bottom:0.4rem; }
.card-badges { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:0.5rem; }
.badge { padding:2px 9px; border-radius:20px; font-size:0.73rem; font-weight:600; }
.badge-cat  { background:#D8F3DC; color:#1B4332; }
.badge-tipe { background:#E8F5E0; color:#2D6A4F; }
.badge-city { background:#F0FFF4; color:#40916C; border:1px solid #B7E4C7; }
.card-meta { display:flex; gap:1rem; font-size:0.82rem; color:#6B8C6A; margin-bottom:0.4rem; }
.card-meta strong { color:#1C2B1A; }
.card-score { display:inline-block; background:#1B4332; color:#fff; font-size:0.73rem; font-weight:600; padding:2px 9px; border-radius:20px; }
.card-desc { font-size:0.81rem; color:#7A9479; line-height:1.55; margin-top:0.4rem; }

.section-title { font-family:'DM Serif Display',serif; font-size:1.5rem; color:#1C2B1A; margin-bottom:0.3rem; }
.section-sub { font-size:0.85rem; color:#8AA88A; margin-bottom:1.2rem; }
.result-banner { background:linear-gradient(90deg,#D8F3DC,#F0FFF4); border-left:4px solid #2D6A4F; border-radius:0 10px 10px 0; padding:0.8rem 1.2rem; margin-bottom:1rem; font-size:0.88rem; color:#1B4332; font-weight:500; }
.empty-state { text-align:center; padding:3rem 2rem; color:#8AA88A; }
.empty-state .icon { font-size:3.5rem; margin-bottom:1rem; }
.empty-state h3 { font-family:'DM Serif Display',serif; color:#2D6A4F; margin-bottom:0.5rem; }
.metric-card { background:#fff; border-radius:14px; padding:1.2rem 1.4rem; box-shadow:0 2px 8px rgba(27,67,50,0.06); border-top:4px solid #52B788; }
.metric-val { font-family:'DM Serif Display',serif; font-size:2rem; color:#1B4332; }
.metric-lbl { font-size:0.78rem; color:#8AA88A; letter-spacing:0.5px; margin-top:2px; }
.metric-model { font-size:0.72rem; color:#52B788; font-weight:600; margin-top:4px; }
.map-legend { background:#fff; border-radius:10px; padding:0.8rem 1.2rem; margin-top:0.5rem; font-size:0.82rem; color:#444; border:1px solid #D8E8D0; line-height:1.7; }

[data-testid="stSidebar"] { background:#1B4332 !important; }
[data-testid="stSidebar"] * { color:#D8F3DC !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color:#fff !important; font-family:'DM Serif Display',serif !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background:rgba(255,255,255,0.1) !important; border-color:rgba(255,255,255,0.2) !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.15) !important; }

.stButton > button { background:#2D6A4F !important; color:#fff !important; border:none !important; border-radius:10px !important; padding:0.55rem 1.2rem !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; font-size:0.9rem !important; width:100%; transition:background 0.2s !important; }
.stButton > button:hover { background:#1B4332 !important; }
.stTextInput > div > div > input { border-radius:10px !important; border:2px solid #D8E8D0 !important; padding:0.55rem 1rem !important; font-size:0.95rem !important; }
.stTextInput > div > div > input:focus { border-color:#2D6A4F !important; box-shadow:0 0 0 3px rgba(45,106,79,0.1) !important; }

/* Expander */
details summary { font-size:0.82rem !important; color:#2D6A4F !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

EMOJI_MAP = {
    'Pantai/Bahari':'🏖️','Gunung/Bukit':'🏔️','Air Terjun':'💧','Goa':'🕳️',
    'Danau/Sungai':'🏞️','Alun-alun':'🌳','Museum':'🏛️','Candi/Keraton':'🛕',
    'Kebun/Taman':'🌿','Budaya':'🎭','Taman Hiburan':'🎡','Bahari':'🌊',
    'Cagar Alam':'🌲','Tempat Ibadah':'🕌','Pusat Perbelanjaan':'🛍️','Lainnya':'📍',
}
KOTA_COORD = {
    'Yogyakarta':(-7.7956,110.3695),
    'Bandung':(-6.9175,107.6191),
    'Jakarta':(-6.2088,106.8456),
    'Surabaya':(-7.2575,112.7521),
    'Semarang':(-6.9932,110.4203),
}

@st.cache_data
def load_data():
    df = pd.read_csv('tourism_preprocessed.csv')
    rating = pd.read_csv('tourism_rating.csv')
    return df, rating

@st.cache_resource
def build_models(df):
    fitur = df['Fitur_Gabungan'].fillna('')
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(fitur)
    cos_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
    count = CountVectorizer(max_features=5000, ngram_range=(1,2))
    count_matrix = count.fit_transform(fitur)
    cos_count = cosine_similarity(count_matrix, count_matrix)
    knn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    return tfidf_matrix, cos_tfidf, cos_count, knn

def rekomendasikan(query, df, cosine_sim, top_k=10):
    matches = df[df['Place_Name'].str.lower() == query.strip().lower()]
    if matches.empty: return pd.DataFrame()
    idx = matches.index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_k+1]
    idxs = [i[0] for i in sim_scores]
    scores = [round(i[1],4) for i in sim_scores]
    hasil = df.iloc[idxs][['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = scores
    return hasil.reset_index(drop=True)

def rekomendasikan_knn(query, df, knn, tfidf_matrix, top_k=10):
    matches = df[df['Place_Name'].str.lower() == query.strip().lower()]
    if matches.empty: return pd.DataFrame()
    idx = matches.index[0]
    distances, idxs = knn.kneighbors(tfidf_matrix[idx], n_neighbors=top_k+1)
    idxs = idxs[0][1:]; distances = distances[0][1:]
    hasil = df.iloc[idxs][['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = [round(1-d,4) for d in distances]
    return hasil.reset_index(drop=True)

def rekomendasikan_by_tipe(tipe, df, top_k=10):
    mask = df['Tipe_Wisata'].str.lower().str.contains(tipe.lower(),na=False) | \
           df['Category'].str.lower().str.contains(tipe.lower(),na=False)
    hasil = df[mask].sort_values('Rating',ascending=False).head(top_k)
    hasil = hasil[['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = None
    return hasil.reset_index(drop=True)

def cari_by_keyword(keyword, df, top_k=10):
    """
    Cari ketat: prioritaskan yang nama/tipe-nya mengandung keyword,
    baru deskripsi. Pastikan semua hasil relevan.
    """
    kw = keyword.strip().lower()

    # Level 1: ada di nama tempat atau tipe wisata (paling relevan)
    mask_utama = (
        df['Place_Name'].str.lower().str.contains(kw, na=False) |
        df['Tipe_Wisata'].str.lower().str.contains(kw, na=False) |
        df['Category'].str.lower().str.contains(kw, na=False) |
        df['City'].str.lower().str.contains(kw, na=False)
    )
    hasil_utama = df[mask_utama].copy()

    # Level 2: ada di deskripsi (kurang relevan, hanya tambahkan jika hasil utama kurang)
    if len(hasil_utama) < top_k:
        mask_desc = df['Description'].str.lower().str.contains(kw, na=False) & ~mask_utama
        hasil_desc = df[mask_desc].copy()
        hasil = pd.concat([hasil_utama, hasil_desc]).head(top_k)
    else:
        hasil = hasil_utama.head(top_k)

    hasil = hasil.sort_values('Rating', ascending=False)
    hasil = hasil[['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = None
    return hasil.reset_index(drop=True)

def apply_filter(hasil, harga_opt, min_rating):
    if hasil is None or hasil.empty: return hasil
    hasil = hasil[hasil['Rating'] >= min_rating]
    if harga_opt == "Gratis": hasil = hasil[hasil['Price'] == 0]
    elif harga_opt == "Berbayar": hasil = hasil[hasil['Price'] > 0]
    return hasil.reset_index(drop=True)

def render_kartu(i, row):
    """Render satu kartu tempat wisata menggunakan st.container (bukan HTML penuh)"""
    tipe  = str(row.get('Tipe_Wisata','Lainnya'))
    cat   = str(row.get('Category','Lainnya'))
    emoji = EMOJI_MAP.get(tipe, EMOJI_MAP.get(cat, '📍'))
    harga = f"Rp {int(row['Price']):,}" if row['Price'] > 0 else "Gratis ✓"
    skor  = f"{row['Skor']:.0%}" if row.get('Skor') is not None and not pd.isna(row['Skor']) else None
    rank  = i + 1
    desc  = str(row['Description'])

    with st.container():
        col_emoji, col_info = st.columns([1, 6])
        with col_emoji:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#D8F3DC,#B7E4C7);
                        border-radius:12px; height:130px; display:flex;
                        align-items:center; justify-content:center; font-size:2.8rem;">
                {emoji}
            </div>""", unsafe_allow_html=True)

        with col_info:
            # Nama + rank
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px;">
                <span style="font-family:'DM Serif Display',serif; font-size:1.1rem; font-weight:bold; color:#1C2B1A;">
                    {row['Place_Name']}
                </span>
                <span style="font-family:'DM Serif Display',serif; font-size:1.6rem; color:#C8E6C0; line-height:1;">
                    #{rank:02d}
                </span>
            </div>""", unsafe_allow_html=True)

            # Badges
            skor_badge = f'<span style="background:#1B4332;color:#fff;padding:2px 9px;border-radius:20px;font-size:0.73rem;font-weight:600;">🎯 {skor} kecocokan</span>' if skor else ''
            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px;">
                <span class="badge badge-cat">{cat}</span>
                <span class="badge badge-tipe">{emoji} {tipe}</span>
                <span class="badge badge-city">📍 {row['City']}</span>
                {skor_badge}
            </div>
            <div style="font-size:0.83rem;color:#6B8C6A;margin-bottom:6px;">
                ⭐ <strong style="color:#1C2B1A;">{row['Rating']}</strong>
                &nbsp;&nbsp; 🎫 <strong style="color:#1C2B1A;">{harga}</strong>
            </div>""", unsafe_allow_html=True)

            # Deskripsi preview (plain text, bukan HTML)
            desc_prev = desc[:180] + "…" if len(desc) > 180 else desc
            st.caption(desc_prev)

        # Expander deskripsi penuh
        if len(desc) > 180:
            with st.expander(f"📖 Baca deskripsi lengkap"):
                st.write(desc)

        st.markdown('<hr style="border:none;border-top:1px solid #F0F4EC;margin:0.3rem 0 0.8rem 0;">', unsafe_allow_html=True)

def tampilkan_semua_kartu(hasil, show_map=False):
    if hasil is None or hasil.empty:
        st.markdown('<div class="empty-state"><div class="icon">🌿</div><h3>Tidak ada hasil</h3><p>Coba ubah kata kunci atau filter</p></div>', unsafe_allow_html=True)
        return

    kota_result = []
    for i, row in hasil.iterrows():
        render_kartu(i, row)
        kota_result.append(row['City'])

    # Peta
    if show_map and kota_result:
        st.markdown("---")
        st.markdown('<div class="section-title">🗺️ Peta Lokasi Rekomendasi</div>', unsafe_allow_html=True)
        st.caption("Titik biru menunjukkan kota lokasi tempat wisata yang direkomendasikan.")

        # Buat satu baris per tempat wisata (bukan per kota)
        coords = []
        for idx, row in hasil.iterrows():
            kota = row['City']
            if kota in KOTA_COORD:
                lat, lon = KOTA_COORD[kota]
                # Tambah sedikit jitter agar titik tidak tumpuk persis
                jitter = (idx * 0.003)
                coords.append({'lat': lat + jitter, 'lon': lon + jitter})

        if coords:
            map_df = pd.DataFrame(coords)
            st.map(map_df, zoom=6)

            # Keterangan per kota
            kota_count = {}
            for k in kota_result:
                kota_count[k] = kota_count.get(k, 0) + 1
            keterangan = " &nbsp;|&nbsp; ".join(
                [f"📍 <strong>{k}</strong>: {v} tempat" for k,v in kota_count.items()]
            )
            st.markdown(f'<div class="map-legend">{keterangan}</div>', unsafe_allow_html=True)


def main():
    df, rating = load_data()
    tfidf_matrix, cos_tfidf, cos_count, knn = build_models(df)

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌿 WisataKu")
        st.markdown("*Sistem Rekomendasi Wisata Indonesia*")
        st.markdown("---")
        st.markdown("### ⚙️ Pengaturan")
        mode = st.radio("Mode Pencarian", [
            "🔍 Ketik Kata Kunci",
            "📋 Pilih dari Daftar",
            "🗂️ Tipe Wisata"
        ])
        metode = st.selectbox("Metode Rekomendasi", [
            "TF-IDF + Cosine Similarity",
            "Count Vectorizer + Cosine Similarity",
            "KNN"
        ])
        top_k = st.slider("Jumlah Rekomendasi", 5, 20, 10)
        show_map = st.toggle("Tampilkan Peta 🗺️", value=True)
        st.markdown("---")
        st.markdown("### 💰 Filter Harga")
        harga_opt = st.radio("", ["Semua","Gratis","Berbayar"], horizontal=True)
        st.markdown("### ⭐ Filter Rating")
        min_rating = st.slider("Rating minimal", 1.0, 5.0, 3.0, 0.1)
        st.markdown("---")
        st.markdown("### 📊 Dataset")
        st.markdown(f"**{len(df)}** tempat wisata")
        st.markdown(f"**{df['City'].nunique()}** kota")
        st.markdown(f"**{df['Category'].nunique()}** kategori")
        st.markdown(f"**{len(rating):,}** data rating")

    # ── HERO ─────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-tag">🌿 Machine Learning · Indonesia</div>
        <h1>Temukan Destinasi<br><em>Wisata Terbaikmu</em></h1>
        <p>Sistem rekomendasi cerdas berbasis TF-IDF & Cosine Similarity untuk menemukan tempat wisata yang paling sesuai untukmu.</p>
        <div class="hero-stats">
            <div><div class="hero-stat-val">437</div><div class="hero-stat-lbl">Tempat Wisata</div></div>
            <div><div class="hero-stat-val">5</div><div class="hero-stat-lbl">Kota</div></div>
            <div><div class="hero-stat-val">10K</div><div class="hero-stat-lbl">Data Rating</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Rekomendasi","📊 Evaluasi Model","📋 Data Wisata"])

    with tab1:

        # MODE 1: Ketik Kata Kunci
        if "Kata Kunci" in mode:
            st.markdown('<div class="section-title">🔍 Cari dengan Kata Kunci</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Ketik nama, tipe, atau kota — contoh: "pantai", "kebun binatang", "Bandung"</div>', unsafe_allow_html=True)

            # Input + tombol sejajar
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                keyword = st.text_input("", placeholder="Ketik kata kunci di sini...", label_visibility="collapsed", key="kw_input")
            with col_btn:
                cari = st.button("🔍 Cari", key="btn_cari")

            if cari and keyword:
                with st.spinner("Mencari..."):
                    hasil = cari_by_keyword(keyword, df, top_k)
                hasil = apply_filter(hasil, harga_opt, min_rating)
                if not hasil.empty:
                    st.markdown(f'<div class="result-banner">✅ Ditemukan <strong>{len(hasil)} tempat wisata</strong> untuk kata kunci "<strong>{keyword}</strong>"</div>', unsafe_allow_html=True)
                    tampilkan_semua_kartu(hasil, show_map=show_map)
                else:
                    st.warning(f'Tidak ditemukan untuk "{keyword}". Coba kata kunci lain atau ubah filter.')
            elif cari and not keyword:
                st.warning("Masukkan kata kunci dulu!")
            else:
                st.markdown('<div class="empty-state"><div class="icon">🔍</div><h3>Ketik Kata Kunci</h3><p>Contoh: "pantai Yogyakarta", "museum Jakarta", "kebun binatang"</p></div>', unsafe_allow_html=True)

        # MODE 2: Pilih dari Daftar
        elif "Daftar" in mode:
            st.markdown('<div class="section-title">📋 Rekomendasi Berdasarkan Tempat</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Pilih satu tempat wisata, sistem akan merekomendasikan tempat serupa</div>', unsafe_allow_html=True)

            search_nama = st.text_input("🔎 Cari nama tempat:", placeholder="Ketik untuk filter daftar...", key="search_nama")
            daftar = df['Place_Name'].tolist()
            if search_nama:
                daftar = [n for n in daftar if search_nama.lower() in n.lower()]
            daftar = sorted(daftar)

            if daftar:
                pilihan = st.selectbox("Pilih dari hasil:", daftar)
                cari2 = st.button("🔍 Rekomendasikan", key="btn_rec")
                if cari2:
                    with st.spinner("Mencari rekomendasi..."):
                        if metode == "TF-IDF + Cosine Similarity":
                            hasil2 = rekomendasikan(pilihan, df, cos_tfidf, top_k)
                        elif metode == "Count Vectorizer + Cosine Similarity":
                            hasil2 = rekomendasikan(pilihan, df, cos_count, top_k)
                        else:
                            hasil2 = rekomendasikan_knn(pilihan, df, knn, tfidf_matrix, top_k)
                    hasil2 = apply_filter(hasil2, harga_opt, min_rating)
                    if not hasil2.empty:
                        st.markdown(f'<div class="result-banner">✅ <strong>{len(hasil2)} rekomendasi</strong> serupa dengan <strong>{pilihan}</strong> — metode: {metode}</div>', unsafe_allow_html=True)
                        tampilkan_semua_kartu(hasil2, show_map=show_map)
                    else:
                        st.warning("Tidak ada hasil sesuai filter.")
                else:
                    st.markdown('<div class="empty-state"><div class="icon">📋</div><h3>Pilih Tempat Wisata</h3><p>Pilih dari daftar, lalu klik Rekomendasikan</p></div>', unsafe_allow_html=True)
            else:
                st.warning("Tidak ada tempat yang cocok dengan pencarianmu.")

        # MODE 3: Tipe Wisata
        else:
            st.markdown('<div class="section-title">🗂️ Rekomendasi Berdasarkan Tipe Wisata</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Pilih tipe wisata yang kamu inginkan</div>', unsafe_allow_html=True)

            tipe_list = sorted(df['Tipe_Wisata'].unique().tolist())
            col1, col2 = st.columns([4,1])
            with col1:
                tipe_pilih = st.selectbox("", tipe_list, label_visibility="collapsed")
            with col2:
                cari3 = st.button("🔍 Tampilkan", key="btn_tipe")

            if cari3:
                with st.spinner("Mencari..."):
                    hasil3 = rekomendasikan_by_tipe(tipe_pilih, df, top_k)
                hasil3 = apply_filter(hasil3, harga_opt, min_rating)
                if not hasil3.empty:
                    st.markdown(f'<div class="result-banner">✅ Menampilkan <strong>{len(hasil3)} tempat wisata</strong> tipe <strong>{tipe_pilih}</strong> dengan rating tertinggi</div>', unsafe_allow_html=True)
                    tampilkan_semua_kartu(hasil3, show_map=show_map)
                else:
                    st.warning("Tidak ada hasil sesuai filter.")
            else:
                st.markdown('<div class="empty-state"><div class="icon">🗂️</div><h3>Pilih Tipe Wisata</h3><p>Pilih tipe wisata lalu klik Tampilkan</p></div>', unsafe_allow_html=True)

    # ── TAB 2: EVALUASI ──────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">📊 Evaluasi Performa Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Perbandingan metrik evaluasi antar model rekomendasi (K=10)</div>', unsafe_allow_html=True)
        try:
            eval_df = pd.read_csv('hasil_evaluasi.csv', index_col=0)
            cols = st.columns(len(eval_df.columns))
            for col, metric in zip(cols, eval_df.columns):
                best_val   = eval_df[metric].max()
                best_model = eval_df[metric].idxmax()
                short      = best_model.replace('+ Cosine Similarity','').strip()
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{best_val:.4f}</div><div class="metric-lbl">{metric}</div><div class="metric-model">🏆 {short}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(eval_df.style.highlight_max(axis=0, color='#D8F3DC').format("{:.4f}"), use_container_width=True)
            st.markdown("<br>**Grafik Perbandingan**")
            st.bar_chart(eval_df)
        except FileNotFoundError:
            st.info("⚠️ Jalankan dulu: `python evaluasi.py`")

    # ── TAB 3: DATA WISATA ───────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">📋 Eksplorasi Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Filter dan jelajahi 437 tempat wisata Indonesia</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: fkota = st.multiselect("🏙️ Kota", sorted(df['City'].unique()))
        with c2: fkat  = st.multiselect("🏷️ Kategori", sorted(df['Category'].unique()))
        with c3: ftipe = st.multiselect("🌿 Tipe", sorted(df['Tipe_Wisata'].unique()))
        fharga  = st.radio("💰 Harga", ["Semua","Gratis","Berbayar"], horizontal=True)
        frating = st.slider("⭐ Rating minimal", 1.0, 5.0, 1.0, 0.1)
        tampil  = df.copy()
        if fkota:  tampil = tampil[tampil['City'].isin(fkota)]
        if fkat:   tampil = tampil[tampil['Category'].isin(fkat)]
        if ftipe:  tampil = tampil[tampil['Tipe_Wisata'].isin(ftipe)]
        if fharga == "Gratis":     tampil = tampil[tampil['Price']==0]
        elif fharga == "Berbayar": tampil = tampil[tampil['Price']>0]
        tampil = tampil[tampil['Rating'] >= frating]
        st.markdown(f"**{len(tampil)}** dari **{len(df)}** tempat wisata")
        st.dataframe(tampil[['Place_Name','Category','Tipe_Wisata','City','Rating','Price']].reset_index(drop=True), use_container_width=True, height=420)
        st.markdown("---")
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Total", len(tampil))
        cc2.metric("Rating Rata-rata", f"{tampil['Rating'].mean():.2f}" if len(tampil) else "—")
        cc3.metric("Gratis", f"{(tampil['Price']==0).sum()}")
        cc4.metric("Kota", tampil['City'].nunique())

if __name__ == '__main__':
    main()