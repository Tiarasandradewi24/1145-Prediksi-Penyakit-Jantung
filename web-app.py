import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model_svm.joblib")
scaler = joblib.load("scaler.joblib")

st.title("Prediksi Penyakit Jantung dengan SVM")
st.write("Masukkan data pasien untuk memprediksi apakah ada risiko penyakit jantung.")

# Form input
age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=50)
sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
cp = st.selectbox("Tipe Nyeri Dada", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Gula Darah Puasa >120 mg/dl", ["Tidak", "Ya"])
restecg = st.selectbox("Hasil EKG Istirahat", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
thalach = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=250, value=150)
exang = st.selectbox("Angina yang Diinduksi Olahraga", ["Tidak", "Ya"])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
slope = st.selectbox("Slope Segmen ST", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Jumlah Pembuluh Darah (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Hasil Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

# Konversi ke angka
sex_num = 1 if sex == "Laki-laki" else 0
cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs_num = 1 if fbs == "Ya" else 0
restecg_num = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
exang_num = 1 if exang == "Ya" else 0
slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[
        age, sex_num, cp_num, trestbps, chol,
        fbs_num, restecg_num, thalach, exang_num,
        oldpeak, slope_num, ca, thal_num
    ]])

    # Transform input menggunakan scaler
    input_scaled = scaler.transform(input_data)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Pasien memiliki risiko penyakit jantung.")
    else:
        st.success("✅ Pasien tidak memiliki risiko penyakit jantung.")