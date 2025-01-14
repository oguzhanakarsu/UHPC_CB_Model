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

# Özellikler ve birimleri
features = {
    "Cement": [270, 1251.2],
    "Silica fume": [0, 433.7],
    "Slag": [0, 375.0],
    "Fly ash": [0, 356.0],
    "Quartz powder": [0, 397.0],
    "Limestone powder": [0, 1058.2],
    "Nano silica": [0, 47.5],
    "Water": [90, 272.6],
    "Fine aggregate": [0, 1502.8],
    "Coarse aggregate": [0, 1195.0],
    "Fiber": [0, 234.0],
    "Superplasticizer": [1.1, 57.0],
    "Temperature": [20, 210],
    "Relative humidity": [50, 100],
    "Age": [7, 365]
}

# Kullanıcı girişlerini al
input_data = {}
for feature, (min_val, max_val) in features.items():
    col1, col2 = st.columns([2, 1])
    with col1:
        slider_val = st.slider(f"{feature} (kg/m³)", min_value=min_val, max_value=max_val, value=(min_val + max_val) / 2)
    with col2:
        input_val = st.number_input(f"Manual {feature} (kg/m³)", min_value=min_val, max_value=max_val, value=slider_val)
    input_data[feature] = input_val

# Tahmin butonu
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    pool = Pool(input_df)
    prediction = model.predict(pool)
    st.success(f"Predicted Compressive Strength (MPa): {prediction[0]:.2f}")

    # SHAP Analizi
    st.subheader("SHAP Analysis")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pool)

    # SHAP Summary Plot
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

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
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    ax.set_zlabel("Compressive Strength (MPa)")
    st.pyplot(fig)
