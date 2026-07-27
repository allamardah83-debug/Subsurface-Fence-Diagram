import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Manager", layout="wide")

st.title("📂 Data Manager")
st.markdown("This page manages all project data for the geotechnical model.")

st.header("Project Information")

project = st.text_input("Project Name")
client = st.text_input("Client")
location = st.text_input("Location")
engineer = st.text_input("Engineer")

st.divider()

st.header("Upload Project Workbook")

uploaded_file = st.file_uploader(
    "Upload Excel Workbook",
    type=["xlsx"]
)

if uploaded_file:

    sheets = pd.read_excel(uploaded_file, sheet_name=None)

    st.success("Workbook Loaded Successfully")

    st.write("### Sheets Found")

    for sheet in sheets.keys():
        st.write("✅", sheet)

    st.divider()

    for name, df in sheets.items():
        with st.expander(name):
            st.dataframe(df, use_container_width=True)

else:

    st.info(
        """
Expected Excel workbook

• Boreholes
• Lithology
• SPT
• Rock Core
• Groundwater

(We'll generate this template automatically later.)
"""
    )
