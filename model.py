import streamlit as st
import pandas as pd
import pickle
import numpy as np
import traceback

def run():
    # Memuat model yang sudah disimpan
    try:
        with open('full_process.pkl', 'rb') as file:
            full_process = pickle.load(file)
        st.success('Model berhasil dimuat!')
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memuat model: {str(e)}")
        st.write("Full traceback:", traceback.format_exc())
        return

    # Input untuk variabel
    square_feet = st.number_input(label='Luas Tanah (sq ft)', min_value=503, max_value=4999)
    st.caption('Masukkan luas tanah rumah dalam satuan kaki persegi (min_value=503, max_value=4999).')

    num_bedrooms = st.selectbox(label='Jumlah Kamar Tidur', options=[1.0, 2.0, 3.0, 4.0, 5.0])
    st.caption('Pilih jumlah kamar tidur rumah.')

    num_bathrooms = st.selectbox(label='Jumlah Kamar Mandi', options=[1.0, 2.0, 3.0])
    st.caption('Pilih jumlah kamar mandi rumah.')

    garage_spaces = st.selectbox(label='Jumlah Garasi', options=[1.0, 2.0])
    st.caption('Pilih jumlah ruang garasi yang tersedia.')

    year_built = st.number_input(label='Tahun Dibangun', min_value=1950, max_value=2022)
    st.caption('Masukkan tahun rumah dibangun (min_value=1950, max_value=2022).')

    lot_size = st.number_input(label='Ukuran Lahan (acres)', min_value=0.100058, max_value=1.995724)
    st.caption('Masukkan ukuran lahan rumah dalam satuan acre(min_value=0.100058, max_value=1.995724).')

    distance_to_city_center = st.number_input(label='Jarak ke Pusat Kota (km)', min_value=1.046227, max_value=49.968165)
    st.caption('Masukkan jarak rumah dari pusat kota dalam satuan kilometer(min_value=1.046227, max_value=49.968165).')

    # Membuat DataFrame dari input
    data_inf = pd.DataFrame({
        'square_feet': [square_feet],
        'num_bedrooms': [num_bedrooms],
        'num_bathrooms': [num_bathrooms],
        'garage_spaces': [garage_spaces],
        'year_built': [year_built],
        'lot_size': [lot_size],
        'distance_to_city_center': [distance_to_city_center]
    })

    # Menampilkan data yang dimasukkan
    st.write("Data yang dimasukkan untuk prediksi:")
    st.write(data_inf)

    # Tombol prediksi
    if st.button(label='Predict'):
        try:
            # Melakukan prediksi harga berdasarkan model
            y_pred_inf = full_process.predict(data_inf)
            st.write(f'Harga Prediksi Rumah: ${y_pred_inf[0]:,.2f}')
        except Exception as e:
            st.error(f'Terjadi kesalahan dalam melakukan prediksi: {str(e)}')
            st.write("Full traceback:", traceback.format_exc())
