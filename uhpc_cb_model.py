import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt

# Sayfa düzenini yatay yapmak
st.set_page_config(layout="wide")

# Başlık ve Açıklama
st.title("UHPC Prediction with CatBoost")
st.write("Bu uygulama, optimize edilmiş CatBoost modeli kullanarak ultra yüksek performanslı betonun basınç dayanımını tahmin eder.")

# Modeli yükle
model = CatBoostRegressor()
model.load_model("optimized_catboost_model.cbm")

# Modelin beklediği sütun adları
expected_columns = [
    "C", "S", "SF", "LP", "QP", "FA", "NS", "W", "Sand", "Gravel", "Fi",
    "SP", "RH", "T", "Age", "SF/C", "SP/C", "C/W", "BM", "A/BM"
]

# Özellikler ve birimleri
features = {
    "Cement": [270.0, 1251.2],
    "Slag": [0.0, 375.0],
    "Silica fume": [0.0, 433.7],
    "Limestone powder": [0.0, 1058.2],
    "Quartz powder": [0.0, 397.0],
    "Fly ash": [0.0, 356.0],
    "Nano silica": [0.0, 47.5],
    "Water": [90.0, 272.6],
    "Fine aggregate": [0.0, 1502.8],
    "Coarse aggregate": [0.0, 1195.0],
    "Fiber": [0.0, 234.0],
    "Superplasticizer": [1.1, 57.0],
    "Relative humidity": [50.0, 100.0],
    "Temperature": [20.0, 210.0],
    "Age": [7.0, 365.0]
}

units = {
    "Cement": "kg/m³", "Slag": "kg/m³", "Silica fume": "kg/m³", "Limestone powder": "kg/m³", "Quartz powder": "kg/m³", "Fly ash": "kg/m³", "Nano silica": "kg/m³", "Water": "kg/m³",
    "Fine aggregate": "kg/m³", "Coarse aggregate": "kg/m³", "Fiber": "kg/m³", "Superplasticizer": "kg/m³", "Relative humidity": "%", "Temperature": "°C", "Age": "days"
}

# Sol sütunda kullanıcı girişlerini al, sağ sütunda PDP grafiği göster
col1, col2 = st.columns(2)

with col1:
    # Kullanıcı girişlerini al
    input_data = {}
    for feature, (min_val, max_val) in features.items():
        unit = units[feature]
        input_val = st.number_input(f"{feature} ({unit})", min_value=float(min_val), max_value=float(max_val), value=(min_val + max_val) / 2, key=f"input_{feature}")
        input_data[feature] = input_val

    # Ek özelliklerin hesaplanması
    input_data["SF/C"] = input_data["Silica fume"] / input_data["Cement"]
    input_data["SP/C"] = input_data["Superplasticizer"] / input_data["Cement"]
    input_data["C/W"] = input_data["Cement"] / input_data["Water"]
    input_data["BM"] = input_data["Cement"] + input_data["Silica fume"] + input_data["Slag"] + input_data["Fly ash"] + input_data["Limestone powder"] + input_data["Nano silica"]
    input_data["A/BM"] = (input_data["Fine aggregate"] + input_data["Coarse aggregate"]) / input_data["BM"]

    # Giriş verisini DataFrame'e dönüştür ve sütun sırasını garanti altına al
    input_df = pd.DataFrame([input_data])
    input_df.columns = expected_columns

    # Tahmin butonu
    if st.button("Predict"):
        try:
            pool = Pool(input_df)
            prediction = model.predict(pool)
            st.success(f"Predicted Compressive Strength (MPa): {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with col2:
    # PDP Grafikleri
    st.subheader("Partial Dependence Plot (PDP)")
    selectable_features = list(features.keys())
    selected_feature = st.selectbox("Select a feature for PDP", selectable_features)
    if selected_feature in features:
        x_values = np.linspace(features[selected_feature][0], features[selected_feature][1], 50)
        input_df_for_pdp = pd.DataFrame([{selected_feature: x, **{f: input_data[f] for f in input_data if f != selected_feature}} for x in x_values])
        input_df_for_pdp.columns = expected_columns
        pool_for_pdp = Pool(input_df_for_pdp)
        y_values = model.predict(pool_for_pdp)

        # 2D PDP Grafiği
        fig_pdp, ax_pdp = plt.subplots(figsize=(10, 5))
        ax_pdp.plot(x_values, y_values, marker='o')
        ax_pdp.set_xlabel(f"{selected_feature} ({units[selected_feature]})")
        ax_pdp.set_ylabel("Compressive Strength (MPa)")
        ax_pdp.grid(True)
        st.pyplot(fig_pdp)
