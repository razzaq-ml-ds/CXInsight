import os
import tempfile

import gradio as gr
import pandas as pd
import plotly.express as px

from src.data_processor import read_raw_data, load_and_clean_data
from src.column_detector import ColumnDetector
from src.sentiment_analyzer import SentimentAnalyzer
from src.analytics import CustomerAnalytics
from src.topic_analyzer import TopicAnalyzer
from src.cloudflare_llm import ask_cloudflare


BG = "#FAF9F5"
PANEL = "#FFFFFF"
TEXT = "#2B2A27"
MUTED = "#85837C"
ACCENT = "#C96442"      # terracotta
POSITIVE = "#7C9A7E"    # sage
NEGATIVE = "#C4624A"    # clay red
NEUTRAL = "#D9C7A3"     # sand
GRID = "rgba(43,42,39,0.07)"

SENTIMENT_COLORS = {
    "Positive": POSITIVE,
    "Negative": NEGATIVE,
    "Neutral": NEUTRAL,
}

CSS_PATH = os.path.join(os.path.dirname(__file__), "static", "styles.css")
with open(CSS_PATH, "r", encoding="utf-8") as css_file:
    CSS = css_file.read()

FORCE_LIGHT_JS = """
() => {
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
}
"""


def style_chart(fig, height=360):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, Helvetica Neue, Arial, sans-serif",
            color=TEXT,
            size=12
        ),
        title=dict(
            font=dict(
                family="Source Serif 4, Georgia, serif",
                size=17,
                color=TEXT
            ),
            x=0.01
        ),
        height=height,
        margin=dict(l=35, r=25, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=MUTED)
        )
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED)
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED)
    )

    return fig


# --------------------------------------------------
# ANALYSIS SERVICE
# --------------------------------------------------

def analyze_file(file_path):

    if not file_path:
        raise gr.Error("Please upload a CSV file.")

    try:
        raw_df = read_raw_data(file_path)

        if raw_df.empty:
            raise gr.Error("The uploaded CSV contains no records.")

        detector = ColumnDetector()

        detected_column, confidence, candidates = (
            detector.detect_feedback_column(raw_df)
        )

        suggested_column = (
            candidates[0]["column"]
            if candidates
            else detected_column
        )

        if suggested_column not in raw_df.columns:
            raise gr.Error(
                "Could not identify the feedback column. "
                "Please check your CSV format."
            )

        df = load_and_clean_data(
            raw_df,
            feedback_column=suggested_column
        )

        if df.empty:
            raise gr.Error("No valid feedback records were found.")

        if len(df) < 2:
            raise gr.Error(
                "At least two valid feedback records are required."
            )

        sentiment_analyzer = SentimentAnalyzer()
        analyzed_df = sentiment_analyzer.analyze_dataframe(df)

        topic_analyzer = TopicAnalyzer(n_clusters=5)
        analyzed_df = topic_analyzer.analyze_dataframe(analyzed_df)

        analytics = CustomerAnalytics(analyzed_df)

        sentiment_counts = analytics.get_sentiment_distribution()
        sentiment_percentages = analytics.get_sentiment_percentages()
        topic_summary = analytics.get_topic_sentiment()

        total = analytics.get_total_feedback()
        rating = analytics.get_average_rating()
        nps = analytics.get_nps()

        positive = sentiment_percentages.get("Positive", 0)
        negative = sentiment_percentages.get("Negative", 0)
        neutral = sentiment_percentages.get("Neutral", 0)

        # KPI values
        kpis = [
            ("Total feedback", f"{total:,}"),
            ("Positive sentiment", f"{positive:.1f}%"),
            ("Negative sentiment", f"{negative:.1f}%"),
            ("Average rating", f"{rating}/5" if rating is not None else "N/A"),
            ("Net promoter score", f"{nps:+.1f}" if nps is not None else "N/A")
        ]

        kpi_html = '<div class="cx-kpi-grid">'

        for label, value in kpis:
            kpi_html += f"""
            <div class="cx-kpi">
                <div class="cx-kpi-label">{label}</div>
                <div class="cx-kpi-value">{value}</div>
            </div>
            """

        kpi_html += "</div>"

        # Overall sentiment chart
        sentiment_df = pd.DataFrame({
            "Sentiment": sentiment_counts.index,
            "Count": sentiment_counts.values
        })

        sentiment_fig = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Count",
            hole=0.62,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            title="Customer sentiment distribution"
        )

        sentiment_fig.update_traces(
            marker=dict(line=dict(color=BG, width=3)),
            textfont=dict(color="#FFFFFF", size=12),
            sort=False
        )

        style_chart(sentiment_fig)

        # Topic-level sentiment chart
        topic_summary = topic_summary.copy()

        topic_summary["Topic"] = topic_summary.apply(
            lambda row: (
                str(row["topic_label"])
                if pd.notna(row.get("topic_label"))
                else f"Topic {row['topic_id']}"
            ),
            axis=1
        )

        topic_plot_df = topic_summary.melt(
            id_vars=["Topic"],
            value_vars=[
                "positive_percentage",
                "negative_percentage",
                "neutral_percentage"
            ],
            var_name="Sentiment",
            value_name="Percentage"
        )

        topic_plot_df["Sentiment"] = (
            topic_plot_df["Sentiment"]
            .str.replace("_percentage", "", regex=False)
            .str.capitalize()
        )

        topic_fig = px.bar(
            topic_plot_df,
            x="Topic",
            y="Percentage",
            color="Sentiment",
            barmode="stack",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment distribution across topics"
        )

        topic_fig.update_layout(
            yaxis_title="Percentage (%)",
            yaxis_range=[0, 100],
            bargap=0.35
        )

        style_chart(topic_fig, height=430)

        # AI context: aggregated statistics, not raw reviews
        context_lines = [
            f"Total feedback: {total}",
            f"Average rating: {rating}",
            f"NPS: {nps}",
            f"Overall sentiment: positive {positive}%, "
            f"negative {negative}%, neutral {neutral}%",
            "Topic-level sentiment:"
        ]

        for _, row in topic_summary.iterrows():
            context_lines.append(
                f"Topic: {row['Topic']}; "
                f"Total: {row['total_feedback']}; "
                f"Positive: {row['positive_percentage']}%; "
                f"Negative: {row['negative_percentage']}%; "
                f"Neutral: {row['neutral_percentage']}%"
            )

        context = "\n".join(context_lines)

        # Feedback Explorer options
        sentiment_options = ["Positive", "Negative", "Neutral"]

        topic_options = (
            topic_summary["Topic"].dropna().astype(str).tolist()
        )

        status = (
            f"Analysis complete. Feedback column: "
            f"{suggested_column} | Detection confidence: {confidence}"
        )

        return (
            kpi_html,
            sentiment_fig,
            topic_fig,
            topic_summary,
            analyzed_df,
            context,
            gr.update(choices=sentiment_options, value=sentiment_options),
            gr.update(choices=topic_options, value=topic_options),
            status
        )

    except gr.Error:
        raise

    except Exception as error:
        raise gr.Error(f"Analysis failed: {error}")



def filter_feedback(df, search, sentiments, topics):

    if df is None:
        raise gr.Error("Please analyze a dataset first.")

    result = df.copy()

    if sentiments is None:
        sentiments = []

    result = result[result["sentiment"].isin(sentiments)]

    if "topic_label" in result.columns and topics:
        result = result[
            result["topic_label"].astype(str).isin([str(topic) for topic in topics])
        ]

    if search and search.strip():
        mask = result["feedback_text"].astype(str).str.contains(
            search.strip(),
            case=False,
            na=False,
            regex=False
        )

        result = result[mask]

    columns = [
        column for column in [
            "feedback_text",
            "sentiment",
            "topic_label",
            "rating",
            "compound_score"
        ]
        if column in result.columns
    ]

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix="cxinsight_feedback_",
        delete=False,
        encoding="utf-8",
        newline=""
    )

    result.to_csv(temp_file.name, index=False)
    temp_file.close()

    return (
        result[columns],
        f"**{len(result):,} matching records**",
        temp_file.name
    )




def generate_ai_answer(question, context):

    if not context:
        raise gr.Error("Analyze a dataset before using CXInsight AI.")

    if not question or not question.strip():
        raise gr.Error("Please enter a question.")

    try:
        answer = ask_cloudflare(
            user_prompt=question.strip(),
            context=context
        )

        return answer

    except Exception as error:
        raise gr.Error(f"AI request failed: {error}")


with gr.Blocks(
    title="CXInsight | Customer Intelligence",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ),
    css=CSS,
    js=FORCE_LIGHT_JS,
    fill_width=True
) as demo:

    analyzed_data_state = gr.State(None)
    ai_context_state = gr.State("")

    # HEADER
    with gr.Row(elem_id="cx-header"):
        with gr.Column(scale=5):
            gr.HTML("""
                <div class="cx-brand-lockup">
                    <span class="cx-brand-dot"></span>
                    <span class="cx-brand-name">CXInsight</span>
                    <span class="cx-brand-caption">Customer intelligence</span>
                </div>
            """, elem_classes="cx-bare")

        with gr.Column(scale=1, min_width=190):
            gr.HTML("""
                <div class="cx-header-action-wrap">
                    <a class="cx-header-action" href="#cxinsight-ai" title="Jump to AI Assistant">
                        Ask CXInsight <span aria-hidden="true">↗</span>
                    </a>
                </div>
            """, elem_classes="cx-bare")

    # LANDING: hero on the left, upload on the right
    with gr.Row(elem_id="cx-top", equal_height=False):
        with gr.Column(scale=6, elem_id="cx-top-left"):
            gr.HTML("""
                <div id="cx-hero">
                    <div class="cx-eyebrow">Customer experience analytics</div>
                    <h1>Listen deeper.<br><span>Understand better.</span></h1>
                    <p>Turn customer feedback into decisions your team can act on.</p>
                </div>
            """, elem_classes="cx-bare")

        with gr.Column(scale=5, elem_id="cx-upload"):
            file_input = gr.File(
                label="Upload customer feedback CSV",
                file_types=[".csv"],
                type="filepath"
            )

            analyze_button = gr.Button(
                "Analyze feedback",
                variant="primary"
            )

            status_output = gr.Markdown(elem_id="cx-status")

    # BUSINESS OVERVIEW
    gr.HTML(
        '<div class="cx-section"><h2>Business overview</h2></div>',
        elem_classes="cx-bare"
    )

    kpi_output = gr.HTML(
        '<div class="cx-subtitle">Upload a dataset to view business metrics.</div>'
    )

    # SENTIMENT
    gr.HTML(
        '<div class="cx-section"><h2>Sentiment analysis</h2></div>',
        elem_classes="cx-bare"
    )

    with gr.Row():
        sentiment_chart = gr.Plot(label="Sentiment distribution")

    # TOPICS
    gr.HTML(
        '<div class="cx-section"><h2>Topic-level sentiment</h2></div>',
        elem_classes="cx-bare"
    )

    topic_chart = gr.Plot(label="Sentiment across topics")

    topic_table = gr.Dataframe(
        label="Topic sentiment summary",
        interactive=False
    )

    # FEEDBACK EXPLORER
    gr.HTML(
        '<div class="cx-section"><h2>Feedback explorer</h2></div>',
        elem_classes="cx-bare"
    )

    gr.Markdown(
        "Search customer reviews and combine filters before applying them."
    )

    with gr.Row():
        search_input = gr.Textbox(
            label="Search within customer reviews",
            placeholder="Try sizing, delivery, quality, refund..."
        )

    with gr.Row():
        sentiment_filter = gr.CheckboxGroup(
            choices=["Positive", "Negative", "Neutral"],
            value=["Positive", "Negative", "Neutral"],
            label="Filter by sentiment"
        )

        topic_filter = gr.CheckboxGroup(
            choices=[],
            label="Filter by topic"
        )

    apply_filters_button = gr.Button(
        "Apply filters",
        variant="primary"
    )

    feedback_count = gr.Markdown()

    feedback_table = gr.Dataframe(
        label="Matching customer feedback",
        interactive=False
    )

    download_output = gr.File(label="Download filtered feedback")

    # AI ASSISTANT
    gr.HTML(
        '<div id="cxinsight-ai" class="cx-section">'
        '<h2>CXInsight AI</h2>'
        '<p class="cx-subtitle">Customer intelligence, explained.</p>'
        '</div>',
        elem_classes="cx-bare"
    )

    gr.Markdown(
        "Ask questions about your analyzed dataset and receive concise, "
        "evidence-based business recommendations."
    )

    ai_question = gr.Textbox(
        label="Ask CXInsight",
        placeholder="What are the biggest areas of customer dissatisfaction?",
        lines=2,
        elem_id="cx-ai-input"
    )

    ai_submit = gr.Button(
        "Generate insights",
        variant="primary"
    )

    ai_answer = gr.Markdown(
        "Your AI-generated insights will appear here after analysis.",
        elem_id="cx-ai-answer"
    )

    gr.HTML("""
        <div class="cx-footer">
            CXInsight &nbsp;·&nbsp; Customer intelligence, made actionable.
        </div>
    """, elem_classes="cx-bare")

    # EVENT CONNECTIONS
    analyze_button.click(
        fn=analyze_file,
        inputs=[file_input],
        outputs=[
            kpi_output,
            sentiment_chart,
            topic_chart,
            topic_table,
            analyzed_data_state,
            ai_context_state,
            sentiment_filter,
            topic_filter,
            status_output
        ]
    )

    apply_filters_button.click(
        fn=filter_feedback,
        inputs=[
            analyzed_data_state,
            search_input,
            sentiment_filter,
            topic_filter
        ],
        outputs=[
            feedback_table,
            feedback_count,
            download_output
        ]
    )

    ai_submit.click(
        fn=generate_ai_answer,
        inputs=[
            ai_question,
            ai_context_state
        ],
        outputs=ai_answer
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )