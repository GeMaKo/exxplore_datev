import streamlit as st
import matplotlib.pyplot as plt

from data import create_figure, get_circle_data, get_isolated_data, get_moon_data, get_xor_data, DATASETS

st.title("Exxplore - Machine Learning Visualized")

left, center, right = st.columns(3)

with left:
    
    data_noise = st.slider("noise", min_value=0.0, max_value=1.0, step=0.05)
    
    dataset_name = st.selectbox("Select dataset", options=DATASETS.keys())
    data_func = DATASETS[dataset_name]
    fig, ax = plt.subplots(1, 1)
    
    X, y = data_func(data_noise)
    create_figure(X, y, ax)
    st.pyplot(fig)

    
    
    