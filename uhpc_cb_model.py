import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor

# Başlık
st.markdown("<h1 style='text-align: center; color: #FFA500;'>UHPC Prediction</h1>", unsafe_allow_html=True)

# CatBoost modelini yükle
model = CatBoostRegressor()
model.load_model("optimized_catboost_model.cbm")

# Yan panelde değişkenler
st.sidebar.header("Feature Values")
cement = st.sidebar.slider("Cement (C)", min_value=270.0, max_value=1251.2, value=737.9, step=1.0)
slag = st.sidebar.slider("Slag (S)", min_value=0.0, max_value=375.0, value=25.1, step=1.0)
silica_fume = st.sidebar.slider("Silica Fume (SF)", min_value=0.0, max_value=433.7, value=136.9, step=1.0)
limestone_powder = st.sidebar.slider("Limestone Powder (LP)", min_value=0.0, max_value=1058.2, value=41.9, step=1.0)
quartz_powder = st.sidebar.slider("Quartz Powder (QP)", min_value=0.0, max_value=397.0, value=33.3, step=1.0)
fly_ash = st.sidebar.slider("Fly Ash (FA)", min_value=0.0, max_value=356.0, value=26.3, step=1.0)
nano_silica = st.sidebar.slider("Nano Silica (NS)", min_value=0.0, max_value=47.5, value=3.6, step=0.1)
water = st.sidebar.slider("Water (W)", min_value=90.0, max_value=272.6, value=179.9, step=1.0)
sand = st.sidebar.slider("Sand", min_value=0.0, max_value=1502.8, value=995.3, step=1.0)
gravel = st.sidebar.slider("Gravel", min_value=0.0, max_value=1195.0, value=154.8, step=1.0)
fiber = st.sidebar.slider("Fiber (Fi)", min_value=0.0, max_value=234.0, value=56.0, step=1.0)
superplasticizer = st.sidebar.slider("Superplasticizer (SP)", min_value=1.1, max_value=57.0, value=30.0, step=0.1)
relative_humidity = st.sidebar.slider("Relative Humidity (RH)", min_value=50.0, max_value=100.0, value=97.9, step=0.1)
temperature = st.sidebar.slider("Temperature (T)", min_value=20.0, max_value=210.0, value=23.9, step=1.0)
age = st.sidebar.slider("Age (days)", min_value=7, max_value=365, value=37, step=1)

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
})

# Tahmin butonu
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Compressive Strength (MPa): {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
