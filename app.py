import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

st.set_page_config(
    page_title="Prediksi Harga Mobil",
    layout="wide"
)
@st.cache_resource
def load_model():

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "Car_sales.xls")

    df = pd.read_csv(data_path)

    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.dropna(inplace=True)

    all_features = ['Engine_size', 'Horsepower', 'Fuel_efficiency', 'Wheelbase', 'Curb_weight']
    target       = 'Price_in_thousands'

    features = [f for f in all_features if f in df.columns]

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, features, df, target

model, features, df, target = load_model()
rata_pasar = df[target].mean()  # rata-rata harga pasar sebagai pembanding

st.title("Sistem Prediksi Harga Mobil")
st.markdown("Masukkan spesifikasi mobil yang anda inginkan, lalu klik")

# LAYOUT DUA KOLOM
col_kiri, col_kanan = st.columns([1, 1], gap="large")

# KOLOM KIRI — Input spesifikasi
with col_kiri:
    st.subheader("Prediksi Harga Mobil")
    st.caption("Geser slider untuk menyesuaikan spesifikasi mobil")

    # Ambil min/max dari dataset agar slider relevan
    engine_size = st.slider(
        "Engine Size (Kapasitas Mesin, Liter)",
        min_value=float(df['Engine_size'].min()),
        max_value=float(df['Engine_size'].max()),
        value=2.0, step=0.1,
    )

    horsepower = st.slider(
        "Horsepower (Tenaga Mesin, HP)",
        min_value=int(df['Horsepower'].min()),
        max_value=int(df['Horsepower'].max()),
        value=150, step=5,
    )

    fuel_efficiency = st.slider(
        "Fuel Efficiency (Efisiensi BBM, MPG)",
        min_value=int(df['Fuel_efficiency'].min()),
        max_value=int(df['Fuel_efficiency'].max()),
        value=28, step=1,
    )

    wheelbase = st.slider(
        "Wheelbase (Jarak Sumbu Roda, inch)",
        min_value=float(df['Wheelbase'].min()),
        max_value=float(df['Wheelbase'].max()),
        value=105.0, step=0.5,
    )

    curb_weight = st.slider(
        "Curb Weight (Berat Kosong, lbs)",
        min_value=int(df['Curb_weight'].min() * 1000),
        max_value=int(df['Curb_weight'].max() * 1000),
        value=3000, step=50,
    )

    # Tombol prediksi — utama
    hitung = st.button("Hitung Harga Mobil", type="primary", use_container_width=True)


# KOLOM KANAN — Hasil prediksi
with col_kanan:
    st.subheader("Perkiraan Harga Mobil")

    if hitung:
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

        # Tampilkan harga prediksi 
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

        # Indikator perbandingan dengan rata-rata pasar
        if selisih > 0:
            st.warning(f"**${selisih:,.2f}k di atas** rata-rata pasar (${rata_pasar:.2f}k)")
        else:
            st.success(f"**${abs(selisih):,.2f}k di bawah** rata-rata pasar (${rata_pasar:.2f}k)")

        # ── Ringkasan spesifikasi input ───────────────────────────────────────
        st.markdown("**Spesifikasi yang Diinputkan:**")
        label_map = {
            'Engine_size'    : 'Variable 1 — Engine Size',
            'Horsepower'     : 'Variable 2 — Horsepower',
            'Fuel_efficiency': 'Variable 3 — Fuel Efficiency',
            'Wheelbase'      : 'Variable 4 — Wheelbase',
            'Curb_weight'    : 'Variable 5 — Curb Weight',
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
        st.info("Atur spesifikasi di sebelah kiri, lalu klik **Hitung Harga Mobil**.")

    st.markdown("---")
    st.markdown(
        """
        <div style="
            background: #e3f2fd;
            border-radius: 10px;
            padding: 14px 18px;
            font-size: 14px;
        ">
            <strong>Sistem ini dibuat oleh:</strong><br>
            Nama : Yasfi Nur Pangestu</em><br>
            NPM : 237006098</em>
        </div>
        """,
        unsafe_allow_html=True
    )
