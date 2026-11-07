
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_processor import load_and_clean_data
from src.sentiment_analyzer import SentimentAnalyzer
from src.analytics import CustomerAnalytics
from src.topic_analyzer import TopicAnalyzer


st.set_page_config(
    page_title="CXInsight | Customer Experience Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("CXInsight")
st.subheader("Customer Experience Intelligence Platform")

st.write(
    "Understand customer feedback through sentiment analysis "
    "and business analytics."
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

        if df.empty:
            st.warning("No valid feedback records found.")
            st.stop()

        st.success("Dataset uploaded and processed successfully!")

        with st.spinner("Analyzing customer feedback..."):
            sentiment_analyzer = SentimentAnalyzer()
            analyzed_df = sentiment_analyzer.analyze_dataframe(df)

            topic_analyzer = TopicAnalyzer(n_clusters=5)
            analyzed_df = topic_analyzer.analyze_dataframe(analyzed_df)

        analytics = CustomerAnalytics(analyzed_df)

        total_feedback = analytics.get_total_feedback()
        average_rating = analytics.get_average_rating()
        nps = analytics.get_nps()
        sentiment_percentages = analytics.get_sentiment_percentages()
        sentiment_counts = analytics.get_sentiment_distribution()

        st.header("Business Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Feedback",
            total_feedback
        )

        col2.metric(
            "Average Rating",
            f"{average_rating}/5"
            if average_rating is not None
            else "N/A"
        )

        col3.metric(
            "Net Promoter Score",
            f"{nps:+.2f}"
            if nps is not None
            else "N/A"
        )

        positive_percentage = sentiment_percentages.get(
            "Positive", 0
        )

        col4.metric(
            "Positive Feedback",
            f"{positive_percentage:.1f}%"
        )

        st.divider()

        st.header("Sentiment Analysis")

        chart_col1, chart_col2 = st.columns(2)

        sentiment_chart_df = (
            sentiment_counts
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        with chart_col1:
            st.subheader("Sentiment Distribution")

            fig = px.pie(
                sentiment_chart_df,
                names="Sentiment",
                values="Count",
                hole=0.45,
                title="Customer Feedback by Sentiment"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with chart_col2:
            st.subheader("Sentiment Percentages")

            percentage_df = (
                sentiment_percentages
                .rename_axis("Sentiment")
                .reset_index(name="Percentage")
            )

            fig = px.bar(
                percentage_df,
                x="Sentiment",
                y="Percentage",
                text="Percentage",
                title="Percentage of Customer Sentiment"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        st.header("Customer Feedback Topics")

        st.write(
            "Discover the recurring themes in customer feedback "
            "using machine learning."
        )

        topic_summary = (
            analyzed_df
            .groupby(["topic_id", "topic_label"])
            .size()
            .reset_index(name="Feedback Count")
        )

        topic_summary["Topic"] = (
            "Topic " + topic_summary["topic_id"].astype(str)
        )

        fig = px.bar(
            topic_summary,
            x="Topic",
            y="Feedback Count",
            color="Topic",
            hover_data=["topic_label"],
            title="Customer Feedback by Discovered Topic"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Discovered Topic Keywords")

        for topic_id, keywords in topic_analyzer.topic_keywords.items():
            st.write(
                f"**Topic {topic_id}:** {', '.join(keywords)}"
            )

        st.header("Customer Feedback Records")

        st.dataframe(
            analyzed_df.head(20),
            use_container_width=True
        )

        st.caption(
            f"Displaying {len(analyzed_df)} processed feedback records."
        )

    except Exception as error:
        st.error(f"Something went wrong: {error}")

else:
    st.info(
        "Upload a customer feedback CSV file using the sidebar "
        "to begin your analysis."
    )
