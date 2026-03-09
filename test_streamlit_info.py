import streamlit as st

def test():
    try:
        st.info("Test &#36;100", unsafe_allow_html=True)
        print("st.info SUPPORTS unsafe_allow_html=True")
    except TypeError:
        print("st.info DOES NOT support unsafe_allow_html=True")

test()
