import pickle
import os
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk menyimpan histori deteksi
def save_detection_history(detection_history, filename="detection_history.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(detection_history, file)

# Fungsi untuk memuat histori deteksi
def load_detection_history(filename="detection_history.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return []

# Fungsi untuk menyimpan session state
def save_session_state(session_state, filename="session_state.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(session_state, file)

# Fungsi untuk memuat session state
def load_session_state(filename="session_state.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return {}

# Initialize session state
def initialize_session_state():
    if 'detection_history' not in st.session_state:
        st.session_state['detection_history'] = load_detection_history()
    if 'session_data' not in st.session_state:
        st.session_state.update(load_session_state())

# Call the function to initialize session state
initialize_session_state()

# Load saved model
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

# Load dataset for display
dataset = pd.read_csv("clean_data.csv")

# Sidebar with option menu
with st.sidebar:
    page = option_menu(
        "Navigasi Sidebar",
        ["Tentang SMS Spam", "Aplikasi Deteksi", "Tabel Dataset"],
        icons=["info-circle", "search", "table"],
        menu_icon="cast",
        default_index=0,
    )

# Halaman Tentang SMS Spam
if page == "Tentang SMS Spam":
    st.title('Tentang SMS Spam')
    st.write("SMS Spam adalah pesan teks yang tidak diinginkan...")
    st.image("emailspam.png", caption="Contoh SMS Spam", use_column_width=True)

# Halaman Aplikasi Deteksi
elif page == "Aplikasi Deteksi":
    st.title('Sistem Deteksi SMS Spam')
    
    sms_text = st.text_area("Masukkan Teks SMS Dibawah Ini")
    if st.button('Cek Deteksi'):
        clean_teks = sms_text.strip()

        if clean_teks == "":
            st.warning("Mohon Masukkan Pesan Teks SMS")
        else:
            predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]))

            if predict_fraud == 0:
                fraud_detection = "SMS NORMAL"
                st.success(fraud_detection)
            elif predict_fraud == 1:
                fraud_detection = "SMS PENIPUAN"
                st.error(fraud_detection)
            elif predict_fraud == 2:
                fraud_detection = "SMS PROMO"
                st.error(fraud_detection)

            # Simpan hasil ke histori
            detection_entry = {
                'text': clean_teks,
                'prediction': fraud_detection
            }
            st.session_state['detection_history'].append(detection_entry)
            save_detection_history(st.session_state['detection_history'])

# Halaman Tabel Dataset
elif page == "Tabel Dataset":
    st.title('Tabel Dataset')
    
    # Tampilkan dataset asli
    st.subheader('Dataset Asli')
    st.dataframe(dataset)
    
    # Tampilkan histori deteksi
    st.subheader('Histori Deteksi SMS')
    if st.session_state['detection_history']:
        # Mengubah histori menjadi DataFrame
        history_df = pd.DataFrame(st.session_state['detection_history'])
        st.dataframe(history_df)
    else:
        st.write("Belum ada histori deteksi.")

# Simpan session state saat aplikasi berhenti
save_session_state(st.session_state)
