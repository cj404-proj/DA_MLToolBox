import streamlit as st

# Hide
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)


st.header("Apply Machine Learning")

md = f"""
The project aims to create a user friendly UI where we can simply upload the dataset and apply machine learning algorithms simply by clicking.\n

The algorithms available in this project are:
* Regression
* Classification
* Clustering

In each of `Regression` and `Classification`, we again have multiple algorithms.

Even before applying those machine learning algorithms, a section names `Analyse` is included.

In the `Analyze` section, you can find the following features.
* Description of dataset
* Finding nulls
* Visualization
    * Uni-variable histogram
    * Bi-variable scatter plot
* Finding outliers
* Encoding
    * Label encoding
    * One hot encoding
* Finding correlation
* Free form analysing

"""

st.markdown(md)