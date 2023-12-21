import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
from urllib.error import URLError

st.set_page_config(page_title="Hysteresis Curve", page_icon="📊")

st.markdown("# Hysteresis Curve")
st.sidebar.header("Hysteresis Curve")
file = st.file_uploader("Pick a file")
plt.plot(file)
plt.show()
