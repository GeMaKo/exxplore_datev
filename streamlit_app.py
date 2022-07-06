import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from data import DATASETS, cv_scores2df, get_train_test_data
from models import fit_estimator, get_classification_report, init_model, models, fit_estimator_with_cv, get_predictions
from plot import create_dataset_figure, create_decision_figure, create_precision_recall_figure, create_roc_figure

mpl.style.use("default")


st.set_page_config(layout="wide")
st.title("Exxplore - Machine Learning Visualized")

# Sidebar with params to choose your dataset
with st.sidebar:
    st.subheader("Choose your dataset")
    
    dataset_name = st.selectbox("Select dataset", options=DATASETS.keys())
    
    data_noise = st.slider("Select noise ratio", min_value=0.05, max_value=1.0, step=0.05, value=0.3)
    data_ratio = st.slider("Select train/test ratio", min_value=0.1, max_value=0.9, step=0.05, value=0.8)
    data_balance = st.slider("Select balance ratio", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
    
    data_func = DATASETS[dataset_name]
    fig, ax = plt.subplots(1, 1)
    
    X, y = data_func(data_noise, data_balance)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = get_train_test_data(X, y, data_ratio)
    
    data_fig = create_dataset_figure(X_train, X_test, y_train, y_test)
    
    st.pyplot(data_fig, clear_figure=True)


# Main Page
st.subheader("Classifier")

# The main page is a sequence of containers
# Each container holds one model and its predictions
# Each container consists of two columns: left and right
# The right column holds the params for the model and the result of cross-validation
# The left column holds the predictions and plots the decision boundary
widget_key = 0
for model_name, model_dict in models.items():
    with st.container():
        st.write(model_name)
        paramter_values = {}
        left, right = st.columns([60, 40])
        with left:
            for parameter_name, properties in model_dict['parameters'].items():
                if properties["type"] == "select_slider":
                    paramter_values[parameter_name] = st.select_slider(
                        parameter_name,
                        options=properties["values"],
                        key=widget_key)
                elif properties["type"] == "checkbox":
                    paramter_values[parameter_name] = st.checkbox(
                        parameter_name,
                        key=widget_key)
                elif properties["type"] == "selection":
                    paramter_values[parameter_name] = st.selectbox(
                        parameter_name,
                        options=properties["values"],
                        key=widget_key)
                else:
                    pass
                widget_key += 1

            scoring_classification_methods = ["f1", "accuracy"]
            estimator = init_model(models[model_name]["estimator"], paramter_values)
            cv_scores =  fit_estimator_with_cv(estimator, X_train, y_train, scoring_classification_methods)   
            df_cv_results = cv_scores2df(cv_scores)
            
            st.caption("Performance measures based on CV on training data:")
            st.dataframe(df_cv_results)
            st.caption(f"Average fit time: {cv_scores['fit_time'].mean(): .5f} seconds")
            st.caption(f"Average score time: {cv_scores['score_time'].mean(): .5f} seconds")
            
            fitted_estimator = fit_estimator(estimator, X_train, y_train)
            models[model_name]["fitted_estimator"] = fitted_estimator
            
            y_pred = get_predictions(fitted_estimator, X_train, y_train)
            models[model_name]["y_pred"] = y_pred
            report = get_classification_report(y_train, y_pred)
            #df_report = pd.DataFrame.from_dict(report)
            st.markdown(f"```\n{report}\n```")
            
            
        with right:
            discretize = st.checkbox("Discretize prediction", key=f"discretize_{widget_key}")
            fig = create_decision_figure(fitted_estimator, X_train, X_test, y_train, y_test, discretize)
            st.pyplot(fig)
            
            test_score = fitted_estimator.score(X_test, y_test)
            st.write(f"Test score is: {test_score}")
            
            widget_key += 1
            
        st.markdown("---")

# Finally, show precision-recall curve and roc curve for all models 
with st.container():
    st.subheader("Model selection")
    left, right = st.columns(2)
    
    with left:
        fig, ax = plt.subplots(1, 1)
        
        for model_name, model_dict in models.items():
            y_pred = model_dict["y_pred"]
            _ = create_precision_recall_figure(y_train, y_pred, name=model_name, ax=ax)
            
        st.pyplot(fig)
        
    with right:
        fig, ax = plt.subplots(1, 1)
        
        for model_name, model_dict in models.items():
            y_pred = model_dict["y_pred"]
            _ = create_roc_figure(y_train, y_pred, name=model_name, ax=ax)
            
        st.pyplot(fig)
        
