# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pages
import plotly.express as px
import plotly.graph_objects as go

# Hide
hide_st_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)


# # Clustering things
# from yellowbrick.cluster import KElbowVisualizer
# from yellowbrick.cluster.elbow import kelbow_visualizer
# from sklearn.cluster import KMeans

# # Page Config
# st.set_page_config(
#     page_title = "Clustering",
#     layout = "centered"
# )

# Header
st.header("Cluster")

# File
file = st.file_uploader(label="Upload your file")

# Processing
if file:
    # Read and display the file
    df = pd.read_csv(file)
    st.table(df.iloc[:10,:])

    # Cluster button
    elbow = st.button(label="Apply Elbow Technique",use_container_width=True)

    # Session state for cluster
    if 'elbow' not in st.session_state:
        st.session_state.elbow = False
    
    # Processing
    if elbow or st.session_state.elbow:
        st.session_state.elbow = True
        elbow_fig = pages.plot_elbow(data_frame=df)
        st.plotly_chart(elbow_fig,theme=None)

        fig2 = px.scatter(df,df.columns[0],df.columns[1])
        st.plotly_chart(fig2)

        with st.form("k"):
            k_input = st.number_input(label="K",min_value=1,max_value=20,step=1)
            cluster = st.form_submit_button(label="Cluster")
        
        if 'cluster' not in st.session_state:
            st.session_state.cluster = False
        
        if cluster or st.session_state.cluster:
            st.session_state.cluster = True

            model = pages.apply_clustering(df,K=k_input)
            fig = pages.plot_clustering(df,model)
            st.plotly_chart(fig,theme=None)

            X = {}
            with st.form(key="cl"):
                for col in df.columns:
                    X[col] = [st.number_input(label=col)]
                    # X.append(st.number_input(label=col))
                if st.form_submit_button("Predict"):
                    st.write(f"The given point belongs to cluster {model.predict(pd.DataFrame(X))[0]}")



        

        
        


    else:
        st.warning("Please cluster the data")



else:
    st.warning("Please upload the file")
