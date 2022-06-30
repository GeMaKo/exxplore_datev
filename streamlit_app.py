import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from plot import create_dataset_figure
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

from data import DATASETS, get_train_test_data
from models import model_dict

mpl.style.use("default")


st.set_page_config(layout="wide")
st.title("Exxplore - Machine Learning Visualized")

#left, right = st.columns([25,75])

with st.sidebar:
    st.subheader("Choose your dataset")
    
    data_noise = st.slider("Select noise", min_value=0.05, max_value=1.0, step=0.05, value=0.3)
    data_ratio = st.slider("Select train/test ratio", min_value=0.1, max_value=0.9, step=0.05, value=0.8)
    
    dataset_name = st.selectbox("Select dataset", options=DATASETS.keys())
    data_func = DATASETS[dataset_name]
    fig, ax = plt.subplots(1, 1)
    
    X, y = data_func(data_noise)
    X_train, X_test, y_train, y_test = get_train_test_data(X, y, data_ratio)
    
    data_fig = create_dataset_figure(X_train, X_test, y_train, y_test)
    
    st.pyplot(data_fig, clear_figure=True)


with st.container():
    left, right = st.columns([60,40])
    with left:
        st.subheader("Classifier")
        paramter_values = {}
        for model_name in model_dict:
            st.write(model_name)
            for parameter_name, properties in model_dict[model_name]['parameters'].items():
                if properties["type"] == "slider":
                    paramter_values[parameter_name] = st.select_slider(
                        parameter_name,
                        options=properties["values"])
                elif properties["type"] == "checkbox":
                    paramter_values[parameter_name] = st.checkbox(
                        parameter_name)
                elif properties["type"] == "selection":
                    paramter_values[parameter_name] = st.selectbox(
                        parameter_name,
                        options=properties["values"])
                else:
                    pass
            model_dict[model_name]["fitted_estimator"] = model_dict[model_name]["estimator"](**paramter_values).fit(X_train, y_train)
            score = model_dict[model_name]["fitted_estimator"].score(X_test, y_test)
            st.write(f"score is: {score}")




    
    
    
    
    
    
    

    
    
    