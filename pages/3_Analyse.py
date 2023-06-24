# Imports
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import io
from streamlit_option_menu import option_menu
import pygwalker as pyg
from sklearn.preprocessing import LabelEncoder

# Hide
# hide_st_style = """
# <style>
# #MainMenu {visibility:hidden;}
# footer {visibility:hidden;}
# header {visibility:hidden;}
# </style>
# """
# st.markdown(hide_st_style,unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title = "Analyze",
    layout = "wide"
)

# Header
st.header("Data Analysis")

# File
file = st.file_uploader(label="Upload your file")

# Processing
if file:
    # Read and display the file
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_excel(file)
    st.table(df.iloc[:5,:])

    # Nav bar
    selected = option_menu(
        menu_title = None,
        options = ["Describe","Nulls","Visualize","Outliers","Encoding","Correlation","Free-Analyze"],
        icons = ["pencil","search","graph-down","exclude"],
        orientation = "horizontal"
    )

    # Navbar sections logic
    if selected == "Describe":
        st.header(f"**Description of the dataset**\n")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.write(pd.DataFrame(df.describe()))

    elif selected == "Nulls":
        st.header("Nulls")
        st.write("**Null values in each column**")
        st.write(df.isna().sum())

    elif selected == "Visualize":
        st.header("Visualize")
        # Create two columns for univariate and bivariate analysis
        uni,bi = st.columns(2)
        with uni:
            feat = st.selectbox(label="Feature",options=df.columns)
            fig = px.histogram(data_frame=df,x=feat)
            st.plotly_chart(fig,use_container_width=True)
        with bi:
            x = st.selectbox(label="X",options=df.columns)
            y = st.selectbox(label="Y",options=df.columns)
            fig = px.scatter(data_frame=df,x=x,y=y)
            st.plotly_chart(fig,use_container_width=True)

    elif selected == "Outliers":
        st.header("Outliers")
        feat = st.selectbox(label="Feature",options=df.columns)
        fig = px.box(data_frame=df,x=feat)
        st.plotly_chart(fig,use_container_width=True)

    elif selected == "Encoding":
        le,ohe,format = st.tabs(["Label Encoding","One Hot Encoding","Format"])
        with le:
            st.header("Label Encoding")
            with st.form("le"):
                st.title("Label Encoding")
                le_col = st.multiselect(label="Select the columns you wish to apply LE",options=df.columns)
                st.form_submit_button(label="Encode")
            st.write(le_col)
            df_c = df.copy()
            label_encoder = LabelEncoder()
            for feat in le_col:
                df_c[feat] = label_encoder.fit_transform(df_c[feat])
            st.write(df_c)
            
        with ohe:
            st.header("One Hot Encoding")
            # df_c = df.copy()
            with st.form("ohe"):
                st.title("One Hot Encoding")
                ohe_col = st.multiselect(label="Select the columns you wish to apply OHE",options=df_c.columns)
                st.form_submit_button(label="Encode")
            st.write(ohe_col)
            df_c = pd.get_dummies(data=df_c,columns=ohe_col)
            st.write(df_c)
        with format:
            st.header("Format")
            with st.form("format"):
                st.title("Format Numeric Column")
                format_col = st.selectbox(label="Columns",options=df_c.columns)
                st.form_submit_button(label="Format")
            st.write(format_col)
            df_c[format_col] = df_c[format_col].replace('\D', '', regex=True).astype(int)
            st.write(df_c)
            csv = df_c.to_csv(index=False)
            st.download_button("Download CSV",data=csv,file_name="df.csv")
    
    elif selected == "Correlation":
        st.header("Correlation")
        corr = df.corr()
        st.write(corr)
        st.plotly_chart(px.imshow(corr,text_auto=True,aspect="auto"),use_container_width=True)
    
    elif selected == "Free-Analyze":
        st.header("Analyze freely")
        pyg.walk(df=df,env="Streamlit")
        
else:
    st.warning("Upload the file")

