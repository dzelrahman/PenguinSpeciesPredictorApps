import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Aplikasi Prediksi Jenis Penguin

Aplikasi ini memprediksi spesies **Penguin Palmer** !!

Data diambil dari ....
""")

st.sidebar.header("Fitur Input User")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/code/master/streamlit/part3/penguins_example.csv)
""")

# memasukkan input dari user ke dalam dataframe
uploaded_file = st.sidebar.file_uploader("Silahkan unggah file CSV Anda", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Pulau", ("Biscoe","Dream","Torgensen"))
        sex = st.sidebar.selectbox("Jenis Kelamin", ("male","female"))
        bill_length_mm = st.sidebar.slider("Panjang paruh (mm)", 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider("Lebar paruh (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Panjang sirip (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Massa tubuh (mm)", 2700.0, 6300.0, 4207.0)
        data = {"island":island,
                "bill_length_mm":bill_length_mm,
                "bill_depth_mm":bill_depth_mm,
                "flipper_length_mm":flipper_length_mm,
                "body_mass_g":body_mass_g,
                "sex":sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# menggabungkan input user dengan keseluruhan dataset penguins
# Berguna untuk fase encoding
penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=["species"])
df = pd.concat([input_df,penguins],axis=0)

# encoding terhadap fitur ordinal (kategori lebih dari 2)
encode = ["sex","island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
df = df[:1] # Hanya pilih baris pertama (data input user)

# Menampilkan fitur input user
st.subheader("Fitur Input User")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Menunggu file CSV berhasil diunggah. Saat ini menggunakan parameter input contoh (terlihat di bawah).")
    st.write(df)

# membaca model klasifikasi yang telah dibuat
load_clf = pickle.load(open("penguins_clf.pkl","rb"))

# terapkan model untuk membuat prediksi
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader("Prediksi")
penguins_species = np.array(["Adelie","Chinstrap","Gentoo"])
st.write(penguins_species[prediction])

st.subheader("Probabilitas / Peluang Prediksi Tiap Kelas")
st.write(prediction_proba)
