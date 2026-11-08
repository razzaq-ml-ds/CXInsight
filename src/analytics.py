
import pandas as pd


class CustomerAnalytics:

    def __init__(self, df):
        self.df = df.copy()

    def get_total_feedback(self):
        return len(self.df)

    def get_sentiment_distribution(self):

        if "sentiment" not in self.df.columns:
            raise ValueError("Sentiment column is missing.")

        return self.df["sentiment"].value_counts()

    def get_sentiment_percentages(self):

        if "sentiment" not in self.df.columns:
            raise ValueError("Sentiment column is missing.")

        return (
            self.df["sentiment"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    def get_average_rating(self):

        if "rating" not in self.df.columns:
            return None

        ratings = pd.to_numeric(
            self.df["rating"],
            errors="coerce"
        ).dropna()

        if ratings.empty:
            return None

        return round(ratings.mean(), 2)

    def get_nps(self):

        if "nps_score" not in self.df.columns:
            return None

        scores = pd.to_numeric(
            self.df["nps_score"],
            errors="coerce"
        ).dropna()

        scores = scores[scores.between(0, 10)]

        if scores.empty:
            return None

        promoters = (scores >= 9).sum()
        detractors = (scores <= 6).sum()
        total = len(scores)

        nps = (
            (promoters / total) -
            (detractors / total)
        ) * 100

        return round(nps, 2)

    def get_topic_sentiment(self):

        required_columns = ["topic_id", "sentiment"]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Required column '{column}' is missing."
                )

        if self.df.empty:
            return pd.DataFrame()

        group_columns = ["topic_id"]

        if "topic_label" in self.df.columns:
            group_columns.append("topic_label")

        summary = (
            self.df.groupby(
                group_columns + ["sentiment"],
                dropna=False
            )
            .size()
            .unstack(fill_value=0)
        )

        for sentiment in ["Positive", "Negative", "Neutral"]:
            if sentiment not in summary.columns:
                summary[sentiment] = 0

        summary = summary[
            ["Positive", "Negative", "Neutral"]
        ]

        summary["total_feedback"] = summary.sum(axis=1)

        for sentiment in ["Positive", "Negative", "Neutral"]:
            summary[f"{sentiment.lower()}_percentage"] = (
                summary[sentiment]
                / summary["total_feedback"]
                * 100
            ).round(2)

        summary = summary.rename(columns={
            "Positive": "positive_count",
            "Negative": "negative_count",
            "Neutral": "neutral_count"
        })

        summary = summary.reset_index()

        summary = summary.sort_values(
            by=["negative_percentage", "negative_count"],
            ascending=False
        ).reset_index(drop=True)

        return summary
