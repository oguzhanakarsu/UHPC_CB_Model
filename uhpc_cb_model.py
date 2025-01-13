import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt

# Başlık
st.markdown("<h1 style='text-align: center; color: #FFA500;'>UHPC Prediction with SHAP Analysis</h1>", unsafe_allow_html=True)

# CatBoost modelini yükle
model = CatBoostRegressor()
model.load_model("optimized_catboost_model.cbm")

# Modelin beklediği sütun adları
expected_columns = [
    "C", "S", "SF", "LP", "QP", "FA", "NS", "W", "Sand", "Gravel", "Fi",
    "SP", "RH", "T", "Age", "SF/C", "SP/C", "C/W", "BM", "A/BM"
]

# Yan panelde değişkenler
st.sidebar.header("Feature Values")

# Kullanıcı girişlerini al
cement = st.sidebar.slider("Cement (C)", 270.0, 1251.2, 737.9, 1.0)
slag = st.sidebar.slider("Slag (S)", 0.0, 375.0, 25.1, 1.0)
silica_fume = st.sidebar.slider("Silica Fume (SF)", 0.0, 433.7, 136.9, 1.0)
limestone_powder = st.sidebar.slider("Limestone Powder (LP)", 0.0, 1058.2, 41.9, 1.0)
quartz_powder = st.sidebar.slider("Quartz Powder (QP)", 0.0, 397.0, 33.3, 1.0)
fly_ash = st.sidebar.slider("Fly Ash (FA)", 0.0, 356.0, 26.3, 1.0)
nano_silica = st.sidebar.slider("Nano Silica (NS)", 0.0, 47.5, 3.6, 0.1)
water = st.sidebar.slider("Water (W)", 90.0, 272.6, 179.9, 1.0)
sand = st.sidebar.slider("Sand", 0.0, 1502.8, 995.3, 1.0)
gravel = st.sidebar.slider("Gravel", 0.0, 1195.0, 154.8, 1.0)
fiber = st.sidebar.slider("Fiber (Fi)", 0.0, 234.0, 56.0, 1.0)
superplasticizer = st.sidebar.slider("Superplasticizer (SP)", 1.1, 57.0, 30.0, 0.1)
relative_humidity = st.sidebar.slider("Relative Humidity (RH)", 50.0, 100.0, 97.9, 0.1)
temperature = st.sidebar.slider("Temperature (T)", 20.0, 210.0, 23.9, 1.0)
age = st.sidebar.slider("Age (days)", 7, 365, 37, 1)

# Yeni özelliklerin hesaplanması
bm = cement + silica_fume + slag + fly_ash + limestone_powder + nano_silica
sf_c_ratio = silica_fume / cement
sp_c_ratio = superplasticizer / cement
c_w_ratio = cement / water
a_bm_ratio = (sand + gravel) / bm

# Kullanıcı girdilerini birleştir
input_data = pd.DataFrame({
    "C": [cement],
    "S": [slag],
    "SF": [silica_fume],
    "LP": [limestone_powder],
    "QP": [quartz_powder],
    "FA": [fly_ash],
    "NS": [nano_silica],
    "W": [water],
    "Sand": [sand],
    "Gravel": [gravel],
    "Fi": [fiber],
    "SP": [superplasticizer],
    "RH": [relative_humidity],
    "T": [temperature],
    "Age": [age],
    "SF/C": [sf_c_ratio],
    "SP/C": [sp_c_ratio],
    "C/W": [c_w_ratio],
    "BM": [bm],
    "A/BM": [a_bm_ratio]
})

# Sütun sırasını garanti altına al
input_data = input_data[expected_columns]

# Girişi CatBoost'un Pool formatına dönüştür
pool = Pool(input_data)

# Tahmin butonu
if st.button("Predict"):
    try:
        prediction = model.predict(pool)
        st.success(f"Predicted Compressive Strength (MPa): {prediction[0]:.2f}")
        
        # SHAP değerlerini hesapla
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pool)

        # SHAP Summary Plot
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(fig)

        # SHAP Waterfall Plot
        st.subheader("SHAP Waterfall Plot")
        shap.waterfall_plot(shap_values[0], max_display=10)
        st.pyplot()
    except Exception as e:
        st.error(f"An error occurred: {e}")
