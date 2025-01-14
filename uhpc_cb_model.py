import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt

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
    "C": [270.0, 1251.2],
    "S": [0.0, 375.0],
    "SF": [0.0, 433.7],
    "LP": [0.0, 1058.2],
    "QP": [0.0, 397.0],
    "FA": [0.0, 356.0],
    "NS": [0.0, 47.5],
    "W": [90.0, 272.6],
    "Sand": [0.0, 1502.8],
    "Gravel": [0.0, 1195.0],
    "Fi": [0.0, 234.0],
    "SP": [1.1, 57.0],
    "RH": [50.0, 100.0],
    "T": [20.0, 210.0],
    "Age": [7.0, 365.0]
}

# Kullanıcı girişlerini al
input_data = {}
for feature, (min_val, max_val) in features.items():
    col1, col2 = st.columns([2, 1])
    with col1:
        slider_val = st.slider(f"{feature} (kg/m³)", min_value=float(min_val), max_value=float(max_val), value=float(min_val), key=f"slider_{feature}")
    with col2:
        input_val = st.number_input(feature, min_value=float(min_val), max_value=float(max_val), value=slider_val, key=f"input_{feature}")
    input_data[feature] = input_val

    # Slider ve number input değerlerini senkronize et
    if st.session_state[f"slider_{feature}"] != st.session_state[f"input_{feature}"]:
        st.session_state[f"slider_{feature}"] = st.session_state[f"input_{feature}"]
        st.session_state[f"input_{feature}"] = st.session_state[f"slider_{feature}"]

# Ek özelliklerin hesaplanması
input_data["SF/C"] = input_data["SF"] / input_data["C"]
input_data["SP/C"] = input_data["SP"] / input_data["C"]
input_data["C/W"] = input_data["C"] / input_data["W"]
input_data["BM"] = input_data["C"] + input_data["SF"] + input_data["S"] + input_data["FA"] + input_data["LP"] + input_data["NS"]
input_data["A/BM"] = (input_data["Sand"] + input_data["Gravel"]) / input_data["BM"]

# Giriş verisini DataFrame'e dönüştür ve sütun sırasını garanti altına al
input_df = pd.DataFrame([input_data])
input_df = input_df[expected_columns]

# Tahmin butonu
if st.button("Predict"):
    pool = Pool(input_df)
    prediction = model.predict(pool)
    st.success(f"Predicted Compressive Strength (MPa): {prediction[0]:.2f}")

    # SHAP Analizi
    st.subheader("SHAP Analysis")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)

    # SHAP Summary Plot
    st.subheader("SHAP Summary Plot")
    fig_summary, ax_summary = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig_summary)

# PDP Grafikleri
st.subheader("Partial Dependence Plot (PDP)")
selected_features = st.multiselect("Select up to 2 features for PDP", list(input_data.keys()), default=list(input_data.keys())[:2])
if len(selected_features) == 2:
    fc_values = np.linspace(20, 150, 50)
    x_values = np.linspace(features[selected_features[0]][0], features[selected_features[0]][1], 50)
    y_values = np.linspace(features[selected_features[1]][0], features[selected_features[1]][1], 50)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.random.uniform(20, 150, X.shape)  # PDP değerleri burada tahminle doldurulabilir

    # 3D PDP Grafiği
    fig_pdp = plt.figure(figsize=(12, 8))
    ax_pdp = fig_pdp.add_subplot(111, projection='3d')
    ax_pdp.plot_surface(X, Y, Z, cmap='viridis')
    ax_pdp.set_xlabel(selected_features[0])
    ax_pdp.set_ylabel(selected_features[1])
    ax_pdp.set_zlabel("Compressive Strength (MPa)")
    st.pyplot(fig_pdp)
