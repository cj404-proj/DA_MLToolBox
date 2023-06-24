# Imports
import numpy as np
import pandas as pd
import streamlit as st
import pages

from sklearn.metrics import classification_report
import io

# Hide
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)

# Page config
# st.set_page_config(
#     page_title = "Classification",
#     layout = "centered"
# )

# Header
st.header("Classification")

# File
file = st.file_uploader(label="Upload your file")

# Processing
if file:
    # Read and display the file
    df = pd.read_csv(file)
    st.table(df.iloc[:10,:])
    # Select target and feature columns
    target_sel_col,feature_sel_col = st.columns(2)
    with target_sel_col:
        target = st.radio(label="Target",options=df.columns)
    with feature_sel_col:
        features = st.multiselect(label="Features",options=df.columns.drop(target))
    # Train button
    train = st.button(label="Train",use_container_width=True)

    # Session state for train
    if 'train' not in st.session_state:
        st.session_state.train = False
    
    # Processing
    if train or st.session_state.train:
        st.session_state.train = True
        # Get Classification Data
        classification_data,models,true,preds = pages.apply_classification(data_frame=df,features=features,target=target)
        # Create tabs
        # linear, dt, svr, lasso, rf, others
        logistic, gnb, mnb, knn, dt, rf, others = st.tabs(["Logistic","GaussinaNB","MultinomialNB","KNN","DT","RF","Others"])
        with logistic:
            st.header("Logistic Regression")
            st.text('Model Report:\n    ' + classification_report(true['logistic'],preds['logistic']))
            st.subheader("Test Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['logistic'])[0])
            st.subheader("Prediction Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['logistic'])[1])
            X = []
            with st.form(key="logistic"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['logistic'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with gnb:
            st.header("Gaussian Naive Bayes")
            st.text('Model Report:\n    ' + classification_report(true['gnb'],preds['gnb']))
            st.subheader("Test Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['gnb'])[0])
            st.subheader("Prediction Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['gnb'])[1])
            X = []
            with st.form(key="gnb"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['gnb'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with mnb:
            st.header("Multi-nomial Naive Bayes")
            st.text('Model Report:\n    ' + classification_report(true['mnb'],preds['mnb']))
            st.subheader("Test Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['mnb'])[0])
            st.subheader("Prediction Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['mnb'])[1])
            X = []
            with st.form(key="mnb"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['mnb'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with knn:
            st.header("KNN")
            st.text('Model Report:\n    ' + classification_report(true['knn'],preds['knn']))
            st.subheader("Test Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['knn'])[0])
            st.subheader("Prediction Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['knn'])[1])
            X = []
            with st.form(key="knn"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['knn'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with dt:
            st.header("Decision Tree")
            st.text('Model Report:\n    ' + classification_report(true['dt'],preds['dt']))
            st.subheader("Test Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['dt'])[0])
            st.subheader("Prediction Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['dt'])[1])
            X = []
            with st.form(key="dt"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['dt'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with rf:
            st.header("Random Forest")
            st.text('Model Report:\n    ' + classification_report(true['rf'],preds['rf']))
            st.subheader("Test Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['rf'])[0])
            st.subheader("Prediction Data")
            st.plotly_chart(pages.plot_clf(df,features,target,models['rf'])[1])
            X = []
            with st.form(key="rf"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['rf'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with others:
            st.header("Other")
            st.dataframe(classification_data)

    else:
        st.warning("Please train the model")
    
else:
    st.warning("Please upload the file")