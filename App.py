import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import base64

st.set_page_config(
    page_title="WisataKu — Rekomendasi Wisata Indonesia",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #F7F9F4; color: #1C2B1A; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1200px; }

.hero {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 45%, #52B788 100%);
    border-radius: 24px; padding: 3rem 3rem; margin: 1.5rem 0 2rem 0;
    position: relative; overflow: hidden;
}
.hero::before { content:''; position:absolute; top:-60px; right:-60px; width:300px; height:300px; background:rgba(255,255,255,0.05); border-radius:50%; }
.hero-tag { display:inline-block; background:rgba(255,255,255,0.15); color:#B7E4C7; font-size:0.75rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; padding:5px 14px; border-radius:20px; margin-bottom:1rem; }
.hero h1 { font-family:'DM Serif Display',serif; font-size:2.8rem; color:#fff; line-height:1.15; margin-bottom:0.8rem; }
.hero h1 em { color:#95D5B2; font-style:italic; }
.hero p { font-size:1rem; color:rgba(255,255,255,0.75); max-width:480px; line-height:1.6; }
.hero-stats { display:flex; gap:2.5rem; margin-top:2rem; }
.hero-stat-val { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#fff; line-height:1; }
.hero-stat-lbl { font-size:0.78rem; color:rgba(255,255,255,0.6); margin-top:2px; }

.setting-bar { background:#fff; border-radius:16px; padding:1.2rem 1.5rem; margin-bottom:1.2rem; box-shadow:0 2px 8px rgba(27,67,50,0.06); border:1px solid #E8F0E4; }
.setting-title { font-size:0.82rem; font-weight:600; color:#2D6A4F; margin-bottom:0.8rem; letter-spacing:0.5px; text-transform:uppercase; }
.filter-box { background:#F0F7EC; border-radius:12px; padding:1rem 1.2rem; margin:0.8rem 0 1rem 0; border:1px solid #D8E8D0; }
.filter-title { font-size:0.78rem; font-weight:600; color:#2D6A4F; margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.5px; }

.stTabs [data-baseweb="tab-list"] { background:transparent; gap:0.5rem; border-bottom:2px solid #D8E8D0; }
.stTabs [data-baseweb="tab"] { font-family:'DM Sans',sans-serif; font-weight:500; font-size:0.9rem; color:#6B8C6A; background:transparent; border-radius:8px 8px 0 0; padding:0.6rem 1.4rem; border:none; }
.stTabs [aria-selected="true"] { color:#1B4332 !important; background:#E8F5E0 !important; border-bottom:3px solid #2D6A4F !important; font-weight:600 !important; }

.section-title { font-family:'DM Serif Display',serif; font-size:1.5rem; color:#1C2B1A; margin-bottom:0.3rem; }
.section-sub { font-size:0.85rem; color:#8AA88A; margin-bottom:1rem; }
.result-banner { background:linear-gradient(90deg,#D8F3DC,#F0FFF4); border-left:4px solid #2D6A4F; border-radius:0 10px 10px 0; padding:0.8rem 1.2rem; margin-bottom:1rem; font-size:0.88rem; color:#1B4332; font-weight:500; }
.empty-state { text-align:center; padding:3rem 2rem; color:#8AA88A; }
.empty-state .icon { font-size:3.5rem; margin-bottom:1rem; }
.empty-state h3 { font-family:'DM Serif Display',serif; color:#2D6A4F; margin-bottom:0.5rem; }

.badge { padding:2px 9px; border-radius:20px; font-size:0.73rem; font-weight:600; }
.badge-cat  { background:#D8F3DC; color:#1B4332; }
.badge-tipe { background:#E8F5E0; color:#2D6A4F; }
.badge-city { background:#F0FFF4; color:#40916C; border:1px solid #B7E4C7; }
.badge-skor { background:#1B4332; color:#fff; }
.card-divider { border:none; border-top:1px solid #F0F4EC; margin:0.3rem 0 0.8rem 0; }

.metric-card { background:#fff; border-radius:14px; padding:1.2rem 1.4rem; box-shadow:0 2px 8px rgba(27,67,50,0.06); border-top:4px solid #52B788; }
.metric-val { font-family:'DM Serif Display',serif; font-size:2rem; color:#1B4332; }
.metric-lbl { font-size:0.78rem; color:#8AA88A; letter-spacing:0.5px; margin-top:2px; }
.metric-model { font-size:0.72rem; color:#52B788; font-weight:600; margin-top:4px; }
.map-legend { background:#fff; border-radius:10px; padding:0.8rem 1.2rem; margin-top:0.5rem; font-size:0.82rem; color:#444; border:1px solid #D8E8D0; }

.stButton > button { background:#2D6A4F !important; color:#fff !important; border:none !important; border-radius:10px !important; padding:0.55rem 1.2rem !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; font-size:0.9rem !important; width:100%; transition:background 0.2s !important; }
.stButton > button:hover { background:#1B4332 !important; }
.stTextInput > div > div > input { border-radius:10px !important; border:2px solid #D8E8D0 !important; padding:0.55rem 1rem !important; font-size:0.95rem !important; }
.stTextInput > div > div > input:focus { border-color:#2D6A4F !important; box-shadow:0 0 0 3px rgba(45,106,79,0.1) !important; }
.stSelectbox > div > div { border-radius:10px !important; border:2px solid #D8E8D0 !important; }
</style>
""", unsafe_allow_html=True)

EMOJI_MAP = {
    'Pantai/Bahari':'🏖️','Gunung/Bukit':'🏔️','Air Terjun':'💧','Goa':'🕳️',
    'Danau/Sungai':'🏞️','Alun-alun':'🌳','Museum':'🏛️','Candi/Keraton':'🛕',
    'Kebun/Taman':'🌿','Budaya':'🎭','Taman Hiburan':'🎡','Bahari':'🌊',
    'Cagar Alam':'🌲','Tempat Ibadah':'🕌','Pusat Perbelanjaan':'🛍️','Lainnya':'📍',
}
KOTA_COORD = {
    'Yogyakarta':(-7.7956,110.3695),'Bandung':(-6.9175,107.6191),
    'Jakarta':(-6.2088,106.8456),'Surabaya':(-7.2575,112.7521),'Semarang':(-6.9932,110.4203),
}

def get_image_path(tipe, cat):
    mapping = {
        'Pantai/Bahari':'images/pantai.jpeg','Bahari':'images/pantai.jpeg',
        'Gunung/Bukit':'images/gunung.jpeg','Cagar Alam':'images/gunung.jpeg',
        'Air Terjun':'images/air_terjun.jpeg','Goa':'images/goa.jpeg',
        'Danau/Sungai':'images/danau.jpeg','Museum':'images/museum.jpeg',
        'Candi/Keraton':'images/candi.jpeg','Tempat Ibadah':'images/candi.jpeg',
        'Kebun/Taman':'images/kebun.jpeg','Budaya':'images/budaya.jpeg',
        'Taman Hiburan':'images/hiburan.jpeg','Pusat Perbelanjaan':'images/hiburan.jpeg',
        'Alun-alun':'images/budaya.jpeg',
    }
    return mapping.get(tipe) or mapping.get(cat) or 'images/pantai.jpeg'

def img_to_base64(path):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

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
    cos_count = cosine_similarity(count.fit_transform(fitur), count.fit_transform(fitur))
    knn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    return tfidf_matrix, cos_tfidf, cos_count, knn

def rekomendasikan(query, df, cosine_sim, top_k):
    matches = df[df['Place_Name'].str.lower() == query.strip().lower()]
    if matches.empty: return pd.DataFrame()
    idx = matches.index[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_k+1]
    idxs = [i[0] for i in sim_scores]; scores = [round(i[1],4) for i in sim_scores]
    hasil = df.iloc[idxs][['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = scores
    return hasil.reset_index(drop=True)

def rekomendasikan_knn(query, df, knn, tfidf_matrix, top_k):
    matches = df[df['Place_Name'].str.lower() == query.strip().lower()]
    if matches.empty: return pd.DataFrame()
    idx = matches.index[0]
    distances, idxs = knn.kneighbors(tfidf_matrix[idx], n_neighbors=top_k+1)
    hasil = df.iloc[idxs[0][1:]][['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = [round(1-d,4) for d in distances[0][1:]]
    return hasil.reset_index(drop=True)

def rekomendasikan_by_tipe(tipe, df, top_k):
    mask = df['Tipe_Wisata'].str.lower().str.contains(tipe.lower(),na=False) | \
           df['Category'].str.lower().str.contains(tipe.lower(),na=False)
    hasil = df[mask].sort_values('Rating',ascending=False).head(top_k)
    hasil = hasil[['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = None
    return hasil.reset_index(drop=True)

def cari_by_keyword(keyword, df, top_k):
    kw = keyword.strip().lower()
    mask_utama = (
        df['Place_Name'].str.lower().str.contains(kw,na=False) |
        df['Tipe_Wisata'].str.lower().str.contains(kw,na=False) |
        df['Category'].str.lower().str.contains(kw,na=False) |
        df['City'].str.lower().str.contains(kw,na=False)
    )
    hasil_utama = df[mask_utama].copy()
    if len(hasil_utama) < top_k:
        mask_desc = df['Description'].str.lower().str.contains(kw,na=False) & ~mask_utama
        hasil = pd.concat([hasil_utama, df[mask_desc]]).head(top_k)
    else:
        hasil = hasil_utama.head(top_k)
    hasil = hasil.sort_values('Rating',ascending=False)
    hasil = hasil[['Place_Name','Category','Tipe_Wisata','City','Rating','Price','Description']].copy()
    hasil['Skor'] = None
    return hasil.reset_index(drop=True)

def apply_filter(hasil, min_rating, filter_kota, price_range):
    if hasil is None or hasil.empty: return hasil
    hasil = hasil[hasil['Rating'] >= min_rating]
    if filter_kota:
        hasil = hasil[hasil['City'].isin(filter_kota)]
    pmin, pmax = price_range
    if pmin == 0:
        hasil = hasil[(hasil['Price'] == 0) | ((hasil['Price'] >= pmin) & (hasil['Price'] <= pmax))]
    else:
        hasil = hasil[(hasil['Price'] >= pmin) & (hasil['Price'] <= pmax)]
    return hasil.reset_index(drop=True)

def render_kartu(i, row):
    tipe  = str(row.get('Tipe_Wisata','Lainnya'))
    cat   = str(row.get('Category','Lainnya'))
    emoji = EMOJI_MAP.get(tipe, EMOJI_MAP.get(cat, '📍'))
    harga = f"Rp {int(row['Price']):,}" if row['Price'] > 0 else "Gratis ✓"
    skor  = row.get('Skor')
    desc  = str(row['Description'])

    img_b64 = img_to_base64(get_image_path(tipe, cat))
    col_img, col_info = st.columns([1, 5])

    with col_img:
        if img_b64:
            st.markdown(f'<div style="border-radius:12px;overflow:hidden;height:140px;"><img src="data:image/jpeg;base64,{img_b64}" style="width:100%;height:100%;object-fit:cover;"/></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:linear-gradient(135deg,#D8F3DC,#B7E4C7);border-radius:12px;height:140px;display:flex;align-items:center;justify-content:center;font-size:3rem;">{emoji}</div>', unsafe_allow_html=True)

    with col_info:
        skor_badge = f'<span class="badge badge-skor">🎯 {skor:.0%} kecocokan</span>' if skor and not pd.isna(skor) else ''
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px;">
            <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;font-weight:bold;color:#1C2B1A;">{row['Place_Name']}</span>
            <span style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#C8E6C0;line-height:1;">#{i+1:02d}</span>
        </div>
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
        st.caption(desc[:180] + "…" if len(desc) > 180 else desc)

    if len(desc) > 180:
        with st.expander("📖 Baca deskripsi lengkap"):
            st.write(desc)
    st.markdown('<hr class="card-divider">', unsafe_allow_html=True)

def tampilkan_kartu(hasil, show_map):
    if hasil is None or hasil.empty:
        st.markdown('<div class="empty-state"><div class="icon">🌿</div><h3>Tidak ada hasil</h3><p>Coba ubah kata kunci atau filter</p></div>', unsafe_allow_html=True)
        return
    kota_result = []
    for i, row in hasil.iterrows():
        render_kartu(i, row)
        kota_result.append(row['City'])

    if show_map and kota_result:
        st.markdown("---")
        st.markdown('<div class="section-title">🗺️ Peta Lokasi</div>', unsafe_allow_html=True)
        st.caption("Setiap titik mewakili satu tempat wisata yang direkomendasikan.")
        coords = []
        for idx, row in hasil.iterrows():
            kota = row['City']
            if kota in KOTA_COORD:
                lat, lon = KOTA_COORD[kota]
                coords.append({'lat': lat+(idx*0.003), 'lon': lon+(idx*0.003)})
        if coords:
            st.map(pd.DataFrame(coords), zoom=6)
            kota_count = {}
            for k in kota_result: kota_count[k] = kota_count.get(k,0)+1
            ket = " &nbsp;|&nbsp; ".join([f"📍 <strong>{k}</strong>: {v} tempat" for k,v in kota_count.items()])
            st.markdown(f'<div class="map-legend">{ket}</div>', unsafe_allow_html=True)

def render_pengaturan(df):
    """Pengaturan dalam halaman (bukan sidebar)"""
    with st.expander("⚙️ Pengaturan & Filter", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.radio("Mode Pencarian", [
                "🔍 Ketik Kata Kunci",
                "📋 Pilih dari Daftar",
                "🗂️ Tipe Wisata"
            ], key="mode")
        with col2:
            metode = st.selectbox("Metode Rekomendasi", [
                "TF-IDF + Cosine Similarity",
                "Count Vectorizer + Cosine Similarity",
                "KNN"
            ], key="metode")
            top_k = st.slider("Jumlah Rekomendasi", 5, 20, 10, key="top_k")
        with col3:
            show_map = st.toggle("Tampilkan Peta 🗺️", value=True, key="show_map")
            min_rating = st.slider("⭐ Rating minimal", 1.0, 5.0, 3.0, 0.1, key="min_rating")

    return mode, metode, top_k, show_map, min_rating

def render_filter(df, key_suffix=""):
    st.markdown('<div class="filter-box">', unsafe_allow_html=True)
    st.markdown('<div class="filter-title">🔽 Filter Hasil</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fkota = st.multiselect("🏙️ Kota", sorted(df['City'].unique()), key=f"fkota{key_suffix}")
    with c2:
        max_price = int(df[df['Price']>0]['Price'].max())
        price_range = st.slider("💰 Rentang Harga (Rp)", 0, max_price, (0, max_price),
                                step=5000, format="Rp %d", key=f"fprice{key_suffix}")
    st.markdown('</div>', unsafe_allow_html=True)
    return fkota, price_range

def main():
    df, rating = load_data()
    tfidf_matrix, cos_tfidf, cos_count, knn = build_models(df)

    # HERO
    n_tempat = len(df)
    n_kota   = df['City'].nunique()
    n_rating = len(rating)
    st.markdown(f"""
    <div class="hero">
        <div class="hero-tag">🌿 Machine Learning · Indonesia</div>
        <h1>Temukan Destinasi<br><em>Wisata Terbaikmu</em></h1>
        <p>Sistem rekomendasi cerdas berbasis TF-IDF & Cosine Similarity untuk menemukan tempat wisata yang paling sesuai untukmu.</p>
        <div class="hero-stats">
            <div><div class="hero-stat-val">{n_tempat}</div><div class="hero-stat-lbl">Tempat Wisata</div></div>
            <div><div class="hero-stat-val">{n_kota}</div><div class="hero-stat-lbl">Kota/Daerah</div></div>
            <div><div class="hero-stat-val">{n_rating:,}</div><div class="hero-stat-lbl">Data Rating</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔍 Rekomendasi","📊 Evaluasi Model","📋 Data Wisata"])

    with tab1:
        # Pengaturan dalam halaman
        mode, metode, top_k, show_map, min_rating = render_pengaturan(df)

        st.markdown("---")

        # ── MODE 1: Kata Kunci ───────────────────────────────
        if "Kata Kunci" in mode:
            st.markdown('<div class="section-title">🔍 Cari dengan Kata Kunci</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Ketik nama, tipe, atau kota — contoh: "pantai", "kebun binatang", "Bandung"</div>', unsafe_allow_html=True)
            col_input, col_btn = st.columns([5,1])
            with col_input:
                keyword = st.text_input("", placeholder="Ketik kata kunci...", label_visibility="collapsed", key="kw_input")
            with col_btn:
                cari = st.button("🔍 Cari", key="btn_cari")

            if cari and keyword:
                st.session_state['hasil_kw'] = cari_by_keyword(keyword, df, top_k)
                st.session_state['keyword_used'] = keyword

            if st.session_state.get('hasil_kw') is not None and not st.session_state['hasil_kw'].empty:
                fkota, price_range = render_filter(df, "_kw")
                hasil = apply_filter(st.session_state['hasil_kw'], min_rating, fkota, price_range)
                kw = st.session_state.get('keyword_used','')
                if not hasil.empty:
                    st.markdown(f'<div class="result-banner">✅ Ditemukan <strong>{len(hasil)} tempat wisata</strong> untuk kata kunci "<strong>{kw}</strong>"</div>', unsafe_allow_html=True)
                    tampilkan_kartu(hasil, show_map)
                else:
                    st.warning("Tidak ada hasil sesuai filter.")
            elif cari and not keyword:
                st.warning("Masukkan kata kunci dulu!")
            else:
                st.markdown('<div class="empty-state"><div class="icon">🔍</div><h3>Ketik Kata Kunci</h3><p>Contoh: "pantai Yogyakarta", "museum Jakarta", "kebun binatang"</p></div>', unsafe_allow_html=True)

        # ── MODE 2: Pilih dari Daftar ────────────────────────
        elif "Daftar" in mode:
            st.markdown('<div class="section-title">📋 Rekomendasi Berdasarkan Tempat</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Pilih satu tempat wisata, sistem akan merekomendasikan tempat serupa</div>', unsafe_allow_html=True)
            search_nama = st.text_input("🔎 Cari nama tempat:", placeholder="Ketik untuk filter daftar...", key="search_nama")
            daftar = sorted([n for n in df['Place_Name'].tolist() if not search_nama or search_nama.lower() in n.lower()])
            if daftar:
                pilihan = st.selectbox("Pilih dari hasil:", daftar)
                cari2 = st.button("🔍 Rekomendasikan", key="btn_rec")
                if cari2:
                    if metode == "TF-IDF + Cosine Similarity":
                        st.session_state['hasil_daftar'] = rekomendasikan(pilihan, df, cos_tfidf, top_k)
                    elif metode == "Count Vectorizer + Cosine Similarity":
                        st.session_state['hasil_daftar'] = rekomendasikan(pilihan, df, cos_count, top_k)
                    else:
                        st.session_state['hasil_daftar'] = rekomendasikan_knn(pilihan, df, knn, tfidf_matrix, top_k)
                    st.session_state['pilihan_used'] = pilihan

                if st.session_state.get('hasil_daftar') is not None and not st.session_state['hasil_daftar'].empty:
                    fkota, price_range = render_filter(df, "_daftar")
                    hasil2 = apply_filter(st.session_state['hasil_daftar'], min_rating, fkota, price_range)
                    if not hasil2.empty:
                        st.markdown(f'<div class="result-banner">✅ <strong>{len(hasil2)} rekomendasi</strong> serupa dengan <strong>{st.session_state.get("pilihan_used","")}</strong></div>', unsafe_allow_html=True)
                        tampilkan_kartu(hasil2, show_map)
                    else:
                        st.warning("Tidak ada hasil sesuai filter.")
                else:
                    st.markdown('<div class="empty-state"><div class="icon">📋</div><h3>Pilih Tempat Wisata</h3><p>Pilih dari daftar, lalu klik Rekomendasikan</p></div>', unsafe_allow_html=True)
            else:
                st.warning("Tidak ada tempat yang cocok.")

        # ── MODE 3: Tipe Wisata ──────────────────────────────
        else:
            st.markdown('<div class="section-title">🗂️ Rekomendasi Berdasarkan Tipe Wisata</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Pilih tipe wisata yang kamu inginkan</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([4,1])
            with col1:
                tipe_pilih = st.selectbox("", sorted(df['Tipe_Wisata'].unique()), label_visibility="collapsed")
            with col2:
                cari3 = st.button("🔍 Tampilkan", key="btn_tipe")

            if cari3:
                st.session_state['hasil_tipe'] = rekomendasikan_by_tipe(tipe_pilih, df, top_k)
                st.session_state['tipe_used'] = tipe_pilih

            if st.session_state.get('hasil_tipe') is not None and not st.session_state['hasil_tipe'].empty:
                fkota, price_range = render_filter(df, "_tipe")
                hasil3 = apply_filter(st.session_state['hasil_tipe'], min_rating, fkota, price_range)
                if not hasil3.empty:
                    st.markdown(f'<div class="result-banner">✅ Menampilkan <strong>{len(hasil3)} tempat wisata</strong> tipe <strong>{st.session_state.get("tipe_used","")}</strong></div>', unsafe_allow_html=True)
                    tampilkan_kartu(hasil3, show_map)
                else:
                    st.warning("Tidak ada hasil sesuai filter.")
            else:
                st.markdown('<div class="empty-state"><div class="icon">🗂️</div><h3>Pilih Tipe Wisata</h3><p>Pilih tipe wisata lalu klik Tampilkan</p></div>', unsafe_allow_html=True)

    # TAB 2: EVALUASI
    with tab2:
        st.markdown('<div class="section-title">📊 Evaluasi Performa Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Perbandingan metrik evaluasi antar model rekomendasi (K=10)</div>', unsafe_allow_html=True)
        try:
            eval_df = pd.read_csv('hasil_evaluasi.csv', index_col=0)
            cols = st.columns(len(eval_df.columns))
            for col, metric in zip(cols, eval_df.columns):
                best_val = eval_df[metric].max()
                best_model = eval_df[metric].idxmax().replace('+ Cosine Similarity','').strip()
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{best_val:.4f}</div><div class="metric-lbl">{metric}</div><div class="metric-model">🏆 {best_model}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(eval_df.style.highlight_max(axis=0, color='#D8F3DC').format("{:.4f}"), use_container_width=True)
            st.markdown("<br>**Grafik Perbandingan**")
            st.bar_chart(eval_df)
        except FileNotFoundError:
            st.info("⚠️ Jalankan dulu: `python evaluasi.py`")

    # TAB 3: DATA WISATA
    with tab3:
        st.markdown('<div class="section-title">📋 Eksplorasi Dataset</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: fkota2 = st.multiselect("🏙️ Kota", sorted(df['City'].unique()), key="fkota2")
        with c2: fkat2  = st.multiselect("🏷️ Kategori", sorted(df['Category'].unique()), key="fkat2")
        with c3: ftipe2 = st.multiselect("🌿 Tipe", sorted(df['Tipe_Wisata'].unique()), key="ftipe2")
        fharga2  = st.radio("💰 Harga", ["Semua","Gratis","Berbayar"], horizontal=True, key="fharga2")
        frating2 = st.slider("⭐ Rating minimal", 1.0, 5.0, 1.0, 0.1, key="frating2")
        tampil = df.copy()
        if fkota2: tampil = tampil[tampil['City'].isin(fkota2)]
        if fkat2:  tampil = tampil[tampil['Category'].isin(fkat2)]
        if ftipe2: tampil = tampil[tampil['Tipe_Wisata'].isin(ftipe2)]
        if fharga2 == "Gratis":     tampil = tampil[tampil['Price']==0]
        elif fharga2 == "Berbayar": tampil = tampil[tampil['Price']>0]
        tampil = tampil[tampil['Rating'] >= frating2]
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