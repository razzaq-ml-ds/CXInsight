
import streamlit as st
import pandas as pd

from src.data_processor import load_and_clean_data
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_analyzer import TopicAnalyzer
from src.analytics import CustomerAnalytics




st.set_page_config(
    page_title="CXInsight | Customer Experience Analytics",
    page_icon="📊",
    layout="wide"
)



st.title("CXInsight")
st.subheader("Customer Experience Intelligence Platform")

st.write(
    "Analyze customer feedback, understand sentiment, "
    "and discover recurring customer experience themes."
)

st.divider()



st.sidebar.header("Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload customer feedback CSV",
    type=["csv"]
)



if uploaded_file is not None:

    try:
      
        df = load_and_clean_data(uploaded_file)

        st.success("CSV uploaded and processed successfully!")

      
        st.subheader("Dataset Preview")

        st.write(
            f"Total records: {len(df)}"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Error processing file: {error}")


else:
    st.info(
        "Please upload a customer feedback CSV file "
        "using the sidebar to begin."
    )
