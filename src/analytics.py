
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
