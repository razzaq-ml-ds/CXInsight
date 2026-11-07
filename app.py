
import streamlit as st
import pandas as pd
import plotly.express as px
from src.cloudflare_llm import ask_cloudflare

from src.data_processor import (
    read_raw_data,
    load_and_clean_data
)

from src.column_detector import ColumnDetector
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

        # STEP 1: Read the original dataset
        raw_df = read_raw_data(uploaded_file)

        if raw_df.empty:
            st.warning("The uploaded CSV contains no records.")
            st.stop()

        if len(raw_df.columns) == 0:
            st.error("The uploaded CSV contains no columns.")
            st.stop()

        # STEP 2: Detect the likely feedback column
        detector = ColumnDetector()

        detected_column, confidence, candidates = (
            detector.detect_feedback_column(raw_df)
        )

        # Use the top candidate as the suggested default
        suggested_column = (
            candidates[0]["column"]
            if candidates
            else None
        )

        st.sidebar.divider()
        st.sidebar.subheader("Feedback Column Selection")

        if confidence == "High":
            st.sidebar.success(
                f"Automatically detected: {suggested_column}"
            )

        elif confidence == "Medium":
            st.sidebar.warning(
                "A possible feedback column was found. "
                "Please confirm your selection."
            )

        else:
            st.sidebar.warning(
                "Could not confidently identify a feedback column. "
                "Please select one manually."
            )

        # STEP 3: Let the user select or override the column
        default_index = (
            raw_df.columns.get_loc(suggested_column)
            if suggested_column is not None
            else 0
        )

        selected_column = st.sidebar.selectbox(
            "Which column contains customer feedback?",
            options=raw_df.columns.tolist(),
            index=default_index,
            help="Select the column containing customer reviews, comments, or feedback.",
            key=f"feedback_column_{uploaded_file.name}"
        )

        st.sidebar.caption(
            f"Selected column: {selected_column}"
        )

        # Show the detector's scoring for transparency
        with st.expander("View column detection results"):

            if candidates:
                candidate_df = pd.DataFrame(candidates)
                st.dataframe(
                    candidate_df,
                    use_container_width=True
                )
            else:
                st.info(
                    "No suitable column candidates were found."
                )

        # STEP 4: Clean the data using the selected column
        df = load_and_clean_data(
            raw_df,
            feedback_column=selected_column
        )

        if df.empty:
            st.warning(
                "No valid feedback records found in the selected column."
            )
            st.stop()

        st.success(
            f"Dataset processed successfully! "
            f"Using '{selected_column}' as the feedback column."
        )

        # STEP 5: Sentiment and topic analysis
        with st.spinner("Analyzing customer feedback..."):

            sentiment_analyzer = SentimentAnalyzer()
            analyzed_df = sentiment_analyzer.analyze_dataframe(df)

            topic_analyzer = TopicAnalyzer(n_clusters=5)
            analyzed_df = topic_analyzer.analyze_dataframe(
                analyzed_df
            )

        analytics = CustomerAnalytics(analyzed_df)

        total_feedback = analytics.get_total_feedback()
        average_rating = analytics.get_average_rating()
        nps = analytics.get_nps()
        sentiment_percentages = analytics.get_sentiment_percentages()
        sentiment_counts = analytics.get_sentiment_distribution()

        # BUSINESS OVERVIEW
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

        # SENTIMENT ANALYSIS
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

        # TOPIC ANALYSIS
        st.header("Customer Feedback Topics")

        st.write(
            "Discover recurring themes in customer feedback "
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
            f"Displaying the first {min(20, len(analyzed_df))} "
            f"of {len(analyzed_df)} processed feedback records."
        )
        
        st.divider()


        st.header("Ask CXInsight")
        st.write(
            "Ask questions about your customer experience analytics "
            "and get AI-powered business insights."
        )

        context_lines = [
            f"Total feedback records: {total_feedback}",
            f"Average rating: {average_rating}",
            f"Net Promoter Score: {nps}",
            f"Sentiment distribution: {sentiment_counts.to_dict()}",
            f"Sentiment percentages: {sentiment_percentages.to_dict()}",
            "",
            "Discovered topics:"
        ]

        for _, row in topic_summary.iterrows():
            topic_id = row["topic_id"]
            keywords = topic_analyzer.topic_keywords.get(
                topic_id, []
            )

            context_lines.append(
                f"Topic {topic_id}: {row['topic_label']}; "
                f"Feedback count: {row['Feedback Count']}; "
                f"Keywords: {', '.join(keywords)}"
            )

        analytics_context = "\n".join(context_lines)

        with st.form("cxinsight_ai_form"):

            user_question = st.text_area(
                "Your question",
                placeholder=(
                    "Example: What are the biggest areas "
                    "where this business should improve?"
                )
            )

            submitted = st.form_submit_button(
                "Generate AI Insights"
            )

        if submitted:

            if not user_question.strip():
                st.warning("Please enter a question.")

            else:
                with st.spinner(
                    "CXInsight is analyzing your business data..."
                ):

                    try:
                        answer = ask_cloudflare(
                            user_prompt=user_question,
                            context=analytics_context
                        )

                        st.subheader("CXInsight's Answer")
                        st.markdown(answer)

                    except Exception as error:
                        st.error(
                            f"Unable to generate insights: {error}"
                        )


    except Exception as error:

        st.error(f"Something went wrong: {error}")


else:

    st.info(
        "Upload a customer feedback CSV file using the sidebar "
        "to begin your analysis."
    )
