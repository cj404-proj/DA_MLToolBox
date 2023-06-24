import streamlit as st
import pages

home, sl, usl = st.tabs(["Home","Supervised Learning","Unsupervised Learning"])

ml_md = pages.ml_md
sl_md = pages.sl_md
lr_sl_md = pages.lr_sl_md
r_sl_md = pages.r_sl_md

with home:
    st.write(ml_md)
with sl:
    st.write(sl_md)
    reg,clf = st.tabs(['Regression','Classification'])
    linear, ridge, lasso, dt, rf = st.tabs(["Linear","Ridge","Lasso","Decision Tree","Random Forest"])
    with linear:
        st.write(lr_sl_md)
    with ridge:
        st.write(r_sl_md)
with usl:
    pass
