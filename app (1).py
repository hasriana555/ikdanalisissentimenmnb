import streamlit as st
import joblib
import re
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.corpus import stopwords
import nltk

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Analisis Sentimen IKD",
    page_icon="📱",
    layout="centered"
)

# =============================================
# LOAD MODEL DAN VECTORIZER
# =============================================
@st.cache_resource
def load_model():
    model = joblib.load('model_svm.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

@st.cache_resource
def load_preprocessing_tools():
    nltk.download('stopwords', quiet=True)

    factory_stem = StemmerFactory()
    stemmer      = factory_stem.create_stemmer()

    factory_stop      = StopWordRemoverFactory()
    sastrawi_sw       = set(factory_stop.get_stop_words())
    nltk_sw           = set(stopwords.words('indonesian'))
    all_stopwords     = sastrawi_sw.union(nltk_sw)

    custom_stopwords = {
        'app', 'apps', 'aplikasi', 'ikd', 'ktp', 'dukcapil',
        'mobile', 'hp', 'android', 'update', 'versi', 'ya',
        'yg', 'aja', 'sy', 'aku', 'ku', 'wae', 'aya', 'tee',
        'mah', 'atuh', 'euy', 'ae', 'rek', 'in', 'log',
        'astagfirullan', 'astaghfirullah', 'astagfirullah',
    }
    all_stopwords = all_stopwords.union(custom_stopwords)

    slang_dict = {
        'gak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak',
        'nggak': 'tidak', 'gk': 'tidak', 'tdk': 'tidak',
        'kga': 'tidak', 'kagak': 'tidak',
        'knp': 'kenapa', 'klo': 'kalau', 'klw': 'kalau',
        'apk': 'aplikasi', 'dlm': 'dalam',
        'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah',
        'blm': 'belum', 'bgt': 'sangat', 'banget': 'sangat',
        'yg': 'yang', 'krn': 'karena', 'karna': 'karena',
        'dgn': 'dengan', 'utk': 'untuk',
        'tp': 'tapi', 'tpi': 'tapi',
        'mau': 'ingin', 'pengen': 'ingin',
        'bs': 'dapat', 'lg': 'lagi',
        'lemot': 'lambat', 'susah': 'sulit', 'ribet': 'rumit',
        'eror': 'error', 'err': 'error',
        'loading': 'muat', 'load': 'muat',
        'download': 'unduh', 'dowload': 'unduh',
        'login': 'masuk', 'bet': 'sangat',
        'gjls': 'tidak jelas', 'gajelas': 'tidak jelas',
    }

    return stemmer, all_stopwords, slang_dict

# =============================================
# FUNGSI PREPROCESSING
# =============================================
def preprocessing(teks, stemmer, all_stopwords, slang_dict):
    if not teks or teks.strip() == '':
        return ''
    teks = teks.lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'@\w+|#\w+', '', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    kata_kata = teks.split()
    teks = ' '.join([slang_dict.get(k, k) for k in kata_kata])
    tokens = teks.split()
    tokens = [k for k in tokens if k not in all_stopwords]
    tokens = [stemmer.stem(k) for k in tokens]
    return ' '.join(tokens)

def konversi_confidence(decision_score):
    """Konversi decision function ke persentase 0-100%"""
    import math
    sigmoid = 1 / (1 + math.exp(-decision_score))
    return sigmoid * 100

# =============================================
# LOAD TOOLS
# =============================================
model, tfidf           = load_model()
stemmer, all_sw, slang = load_preprocessing_tools()

# =============================================
# TAMPILAN
# =============================================
st.title("📱 Analisis Sentimen Ulasan Aplikasi IKD")
st.markdown(
    "Prediksi sentimen ulasan pengguna aplikasi "
    "**Identitas Kependudukan Digital (IKD)** menggunakan "
    "algoritma **Support Vector Machine (SVM)**."
)

st.divider()

# ===== INPUT =====
st.subheader("📝 Input Ulasan")
ulasan = st.text_area(
    label       = "Masukkan ulasan pengguna aplikasi IKD:",
    placeholder = "Contoh: Aplikasi sangat membantu, tidak perlu bawa KTP fisik...",
    height      = 150
)

col1, col2 = st.columns([1, 4])
with col1:
    prediksi_btn = st.button("🔍 Prediksi", type="primary", use_container_width=True)
with col2:
    reset_btn = st.button("🔄 Reset", use_container_width=True)

if reset_btn:
    st.rerun()

# ===== HASIL PREDIKSI =====
if prediksi_btn:
    if ulasan.strip() == '':
        st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu.")
    else:
        with st.spinner('Memproses ulasan...'):
            teks_bersih = preprocessing(ulasan, stemmer, all_sw, slang)

            if teks_bersih.strip() == '':
                st.error("❌ Teks tidak dapat diproses. Coba masukkan ulasan yang lebih lengkap.")
            else:
                teks_tfidf     = tfidf.transform([teks_bersih])
                hasil          = model.predict(teks_tfidf)[0]
                decision_score = model.decision_function(teks_tfidf)[0]

                # Konversi ke confidence score
                if hasil == 'Positif':
                    conf_positif = konversi_confidence(decision_score)
                    conf_negatif = 100 - conf_positif
                else:
                    conf_negatif = konversi_confidence(abs(decision_score))
                    conf_positif = 100 - conf_negatif

                conf_utama = conf_positif if hasil == 'Positif' else conf_negatif

                st.divider()
                st.subheader("📊 Hasil Prediksi")

                # Hasil utama
                if hasil == 'Positif':
                    st.success(f"✅ **Sentimen: POSITIF**")
                else:
                    st.error(f"❌ **Sentimen: NEGATIF**")

                # Confidence score utama
                st.markdown(f"**Confidence Score: {conf_utama:.1f}%**")
                st.progress(int(conf_utama))

                # Visualisasi probabilitas kedua kelas
                st.markdown("**Distribusi Sentimen:**")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("🟢 **Positif**")
                    st.progress(int(conf_positif))
                    st.caption(f"{conf_positif:.1f}%")

                with col2:
                    st.markdown("🔴 **Negatif**")
                    st.progress(int(conf_negatif))
                    st.caption(f"{conf_negatif:.1f}%")

                # Detail preprocessing
                with st.expander("🔍 Lihat detail preprocessing"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Teks asli:**")
                        st.info(ulasan)
                    with col2:
                        st.markdown("**Setelah preprocessing:**")
                        st.info(teks_bersih if teks_bersih else "(kosong)")

st.divider()

# ===== INFO MODEL =====
st.subheader("📈 Performa Model SVM")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi",  "89.36%")
col2.metric("Presisi",  "90.28%")
col3.metric("Recall",   "81.59%")
col4.metric("F1-Score", "85.71%")

st.divider()

# ===== INFO DATASET =====
st.subheader("📂 Informasi Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Data",   "3.055")
col2.metric("Positif",      "1.195")
col3.metric("Negatif",      "1.860")

st.caption(
    "Data diambil dari Google Play Store aplikasi IKD "
    "(gov.dukcapil.mobile_id) menggunakan teknik web scraping."
)

st.divider()

st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px'>"
    "Penelitian Skripsi | Algoritma: SVM | Fitur: TF-IDF"
    "</div>",
    unsafe_allow_html=True
)
