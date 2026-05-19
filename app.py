import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    layout="wide"
)

# ── Load & cache data + model (hanya dijalankan sekali) ───────────────────────
@st.cache_resource
def load_model():
    """
    Fungsi ini membaca dataset, membersihkan data, lalu melatih model.
    @st.cache_resource → hanya dijalankan sekali, hasilnya disimpan di cache
    agar aplikasi tidak lambat setiap kali user berinteraksi.
    """
    # Cari file dataset di folder yang sama dengan app.py
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "Car_sales.xls")

    # Baca dataset
    df = pd.read_csv(data_path)

    # ── Preprocessing: isi missing value ─────────────────────────────────────
    # Kolom numerik → diisi median (lebih robust terhadap outlier)
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Kolom kategorik → diisi modus (nilai yang paling sering muncul)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Hapus sisa baris yang masih NaN (jika ada)
    df.dropna(inplace=True)

    # ── Definisi fitur dan target ─────────────────────────────────────────────
    all_features = ['Engine_size', 'Horsepower', 'Fuel_efficiency', 'Wheelbase', 'Curb_weight']
    target       = 'Price_in_thousands'

    # Hanya pakai fitur yang benar-benar ada di dataset
    features = [f for f in all_features if f in df.columns]

    # ── Training model Linear Regression ─────────────────────────────────────
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, features, df, target

# Panggil fungsi load model
model, features, df, target = load_model()
rata_pasar = df[target].mean()  # rata-rata harga pasar sebagai pembanding

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("Sistem Prediksi Harga Mobil")
st.markdown("Masukkan spesifikasi mobil yang ingin diproduksi, lalu klik **Hitung Harga Mobil**.")
st.markdown("---")

# ── LAYOUT DUA KOLOM ─────────────────────────────────────────────────────────
col_kiri, col_kanan = st.columns([1, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# KOLOM KIRI — Input spesifikasi
# ════════════════════════════════════════════════════════════════════════════
with col_kiri:
    st.subheader("Prediksi Harga Mobil")
    st.caption("Geser slider untuk menyesuaikan spesifikasi mobil")

    # Ambil min/max dari dataset agar slider relevan
    engine_size = st.slider(
        "Variable 1 — Engine Size (Kapasitas Mesin, Liter)",
        min_value=float(df['Engine_size'].min()),
        max_value=float(df['Engine_size'].max()),
        value=2.0, step=0.1,
        help="Kapasitas mesin dalam liter. Semakin besar → umumnya harga lebih tinggi."
    )

    horsepower = st.slider(
        "Variable 2 — Horsepower (Tenaga Mesin, HP)",
        min_value=int(df['Horsepower'].min()),
        max_value=int(df['Horsepower'].max()),
        value=150, step=5,
        help="Tenaga mesin dalam Horse Power. Berpengaruh besar pada segmen harga."
    )

    fuel_efficiency = st.slider(
        "Variable 3 — Fuel Efficiency (Efisiensi BBM, MPG)",
        min_value=int(df['Fuel_efficiency'].min()),
        max_value=int(df['Fuel_efficiency'].max()),
        value=28, step=1,
        help="Jarak tempuh per galon bahan bakar. Semakin tinggi = lebih irit."
    )

    wheelbase = st.slider(
        "Variable 4 — Wheelbase (Jarak Sumbu Roda, inch)",
        min_value=float(df['Wheelbase'].min()),
        max_value=float(df['Wheelbase'].max()),
        value=105.0, step=0.5,
        help="Jarak antara sumbu roda depan dan belakang. Mencerminkan ukuran kendaraan."
    )

    curb_weight = st.slider(
        "Variable 5 — Curb Weight (Berat Kosong, lbs)",
        min_value=int(df['Curb_weight'].min() * 1000),
        max_value=int(df['Curb_weight'].max() * 1000),
        value=3000, step=50,
        help="Berat kendaraan tanpa penumpang/muatan dalam lbs."
    )

    # Tombol prediksi — utama
    hitung = st.button("Hitung Harga Mobil", type="primary", use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# KOLOM KANAN — Hasil prediksi
# ════════════════════════════════════════════════════════════════════════════
with col_kanan:
    st.subheader("Perkiraan Harga Mobil")

    if hitung:
        # ── Siapkan input dan prediksi ────────────────────────────────────────
        nilai_input = {
            'Engine_size'    : engine_size,
            'Horsepower'     : horsepower,
            'Fuel_efficiency': fuel_efficiency,
            'Wheelbase'      : wheelbase,
            'Curb_weight'    : curb_weight / 1000,  # konversi lbs → ribuan untuk konsistensi skala
        }
        input_data = {f: [nilai_input[f]] for f in features}
        df_input   = pd.DataFrame(input_data)

        # Prediksi harga menggunakan model
        harga = model.predict(df_input)[0]
        selisih = harga - rata_pasar

        # ── Tampilkan harga prediksi dengan styling besar ─────────────────────
        st.markdown(
            f"""
            <div style="
                background: #fffde7;
                border: 2px solid #f9a825;
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                margin-bottom: 16px;
            ">
                <div style="font-size: 14px; color: #777; margin-bottom: 4px;">Perkiraan Harga</div>
                <div style="font-size: 48px; font-weight: bold; color: #e65100;">
                    ${harga:,.2f}k
                </div>
                <div style="font-size: 16px; color: #555;">
                    Setara <strong>${harga * 1000:,.0f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Indikator perbandingan dengan rata-rata pasar (pakai HTML agar tidak ada bintang)
        if selisih > 0:
            st.markdown(
                f'<div style="background:#fff3cd;border-left:4px solid #f9a825;padding:10px 14px;'
                f'border-radius:6px;color:#856404;font-size:15px;">'
                f'<b>${selisih:,.2f}k di atas</b> rata-rata pasar (${rata_pasar:.2f}k)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background:#d4edda;border-left:4px solid #28a745;padding:10px 14px;'
                f'border-radius:6px;color:#155724;font-size:15px;">'
                f'<b>${abs(selisih):,.2f}k di bawah</b> rata-rata pasar (${rata_pasar:.2f}k)</div>',
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Ringkasan spesifikasi input ───────────────────────────────────────
        st.markdown("**Spesifikasi yang Diinputkan:**")
        label_map = {
            'Engine_size'    : 'Engine Size',
            'Horsepower'     : 'Horsepower',
            'Fuel_efficiency': 'Fuel Efficiency',
            'Wheelbase'      : 'Wheelbase',
            'Curb_weight'    : 'Curb Weight',
        }
        unit_map = {
            'Engine_size': 'L', 'Horsepower': 'HP',
            'Fuel_efficiency': 'MPG', 'Wheelbase': 'inch', 'Curb_weight': 'ribuan lbs'
        }
        for f in features:
            val = nilai_input[f]
            st.markdown(f"- **{label_map.get(f, f)}** : `{val} {unit_map.get(f,'')}`")

        # ── Chart bar prediksi vs rata-rata ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 2.5))
        bars = ax.barh(
            ['Prediksi Harga', 'Rata-rata Pasar'],
            [harga, rata_pasar],
            color=['#2ecc71', '#f39c12'],
            edgecolor='white', height=0.45
        )
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'${w:,.1f}k', va='center', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(harga, rata_pasar) * 1.25)
        ax.set_xlabel('Harga (ribuan USD)')
        ax.set_title('Prediksi vs Rata-rata Pasar', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        # Tampilan awal sebelum tombol ditekan
        st.info("Atur spesifikasi di sebelah kiri, lalu tekan **Hitung Harga Mobil**.")

    # ── Info pembuat ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: #1565c0;
            border-radius: 10px;
            padding: 16px 20px;
            font-size: 15px;
            color: #ffffff;
        ">
            <strong>Sistem ini dibuat oleh:</strong><br><br>
            Nama &nbsp;: <span style="color:#90caf9;">Yasfi Nur Pangestu</span><br>
            NPM &nbsp;&nbsp;: <span style="color:#90caf9;">237006098</span>
        </div>
        """,
        unsafe_allow_html=True
    )
