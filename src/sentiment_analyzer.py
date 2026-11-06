
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """
    Analyzes customer feedback using VADER sentiment analysis.
    """

    def __init__(self):
        # Initialize VADER analyzer once
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text):
        """
        Analyze a single customer feedback text.
        Returns sentiment scores and classification.
        """

        # Handle missing or empty text
        if pd.isna(text) or not str(text).strip():
            return {
                "positive_score": 0.0,
                "negative_score": 0.0,
                "neutral_score": 0.0,
                "compound_score": 0.0,
                "sentiment": "Neutral"
            }

        # Calculate sentiment scores
        scores = self.analyzer.polarity_scores(str(text))

        compound = scores["compound"]

        # Classify sentiment
        if compound >= 0.05:
            sentiment = "Positive"

        elif compound <= -0.05:
            sentiment = "Negative"

        else:
            sentiment = "Neutral"

        return {
            "positive_score": scores["pos"],
            "negative_score": scores["neg"],
            "neutral_score": scores["neu"],
            "compound_score": compound,
            "sentiment": sentiment
        }

    def analyze_dataframe(self, df):
        """
        Apply sentiment analysis to every feedback record.
        Returns a new DataFrame with sentiment columns.
        """

        # Validate input
        if "feedback_text" not in df.columns:
            raise ValueError(
                "DataFrame must contain feedback_text column."
            )

        # Create a copy to preserve original DataFrame
        result = df.copy()

        # Apply analyze_text() to every feedback entry
        sentiment_results = result["feedback_text"].apply(
            self.analyze_text
        )

        # Convert list of dictionaries into a DataFrame
        sentiment_df = pd.DataFrame(
            sentiment_results.tolist(),
            index=result.index
        )

        # Combine original data with sentiment results
        result = pd.concat(
            [result, sentiment_df],
            axis=1
        )

        return result
