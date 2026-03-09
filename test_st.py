import streamlit as st
try:
    st.info("testing &#36;100", unsafe_allow_html=True)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
