# Imports
import numpy as np
import pandas as pd
import streamlit as st
import pages

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
#     page_title = "Regression",
#     layout = "centered"
# )

# Header
st.header("Regression")

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
    train = st.button(label="TRAIN",use_container_width=True)

    # Session state for train
    if "train" not in st.session_state:
        st.session_state.train = False
    
    # Processing
    if train or st.session_state.train:
        st.session_state.train = True
        # Get Regression Data
        regression_data,models = pages.apply_regression(data_frame=df,features=features,target=target)
        # Create tabs
        linear, dt, svr, lasso, rf, others = st.tabs(["Linear","Decision Tree","Support Vector","Lasso","Random Forest","Others"])
        with linear:
            st.header("Linear Regression")
            st.write(f"{regression_data['LinearRegression']}")
            st.write(pd.DataFrame(data=regression_data['LinearRegression'].values(),index=regression_data['LinearRegression'].keys(),columns=['Value']))
            st.plotly_chart(pages.plot_reg(df,features,target,models['linear']))
            X = []
            with st.form(key="linear"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['linear'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")

        with dt:
            st.header("Decision Tree")
            st.write(f"{regression_data['DecisionTreeRegressor']}")
            st.write(pd.DataFrame(data=regression_data['DecisionTreeRegressor'].values(),index=regression_data['DecisionTreeRegressor'].keys(),columns=['Value']))
            st.plotly_chart(pages.plot_reg(df,features,target,models['dt']))
            X = []
            with st.form(key="dt"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['dt'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with svr:
            st.header("Support Vector Regression")
            st.write(f"{regression_data['SVR']}")
            st.write(pd.DataFrame(data=regression_data['SVR'].values(),index=regression_data['SVR'].keys(),columns=['Value']))
            st.plotly_chart(pages.plot_reg(df,features,target,models['svr']))
            X = []
            with st.form(key="svr"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['svr'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with lasso:
            st.header("Lasso Regression")
            st.write(f"{regression_data['Lasso']}")
            st.write(pd.DataFrame(data=regression_data['Lasso'].values(),index=regression_data['Lasso'].keys(),columns=['Value']))
            st.plotly_chart(pages.plot_reg(df,features,target,models['lasso']))
            X = []
            with st.form(key="lasso"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['lasso'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with rf:
            st.header("Random Forest")
            st.write(f"{regression_data['RandomForestRegressor']}")
            st.write(pd.DataFrame(data=regression_data['RandomForestRegressor'].values(),index=regression_data['RandomForestRegressor'].keys(),columns=['Value']))
            st.plotly_chart(pages.plot_reg(df,features,target,models['rf']))
            X = []
            with st.form(key="rf"):
                for col in features:
                    X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The prediction is {models['rf'].predict(pd.DataFrame(np.array([X]),columns=features))[0]}")
        with others:
            st.header("Other")
            st.dataframe(regression_data)
    else:
        st.warning("Please train the model")


else:
    st.warning("Please upload the file")