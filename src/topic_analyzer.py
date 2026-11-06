
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class TopicAnalyzer:


    def __init__(self, n_clusters=5, max_features=1000):
        self.n_clusters = n_clusters
        self.max_features = max_features

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=(1, 2)
        )

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )

        self.topic_keywords = {}

    def analyze_dataframe(self, df):


        if "feedback_text" not in df.columns:
            raise ValueError(
                "DataFrame must contain feedback_text column."
            )

        result = df.copy()

        texts = (
            result["feedback_text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )


        valid_mask = texts.ne("")

        if valid_mask.sum() < 2:
            raise ValueError(
                "At least two non-empty feedback records are required."
            )


        tfidf_matrix = self.vectorizer.fit_transform(
            texts[valid_mask]
        )

        actual_clusters = min(
            self.n_clusters,
            valid_mask.sum()
        )

        self.model.set_params(n_clusters=actual_clusters)

        cluster_ids = self.model.fit_predict(tfidf_matrix)

        feature_names = self.vectorizer.get_feature_names_out()

        self.topic_keywords = {}

        for cluster_id in range(actual_clusters):

            center = self.model.cluster_centers_[cluster_id]

            top_indices = center.argsort()[-5:][::-1]


            keywords = [
                feature_names[index]
                for index in top_indices
            ]

            self.topic_keywords[cluster_id] = keywords

        result["topic_id"] = pd.NA
        result["topic_label"] = pd.NA

        result.loc[valid_mask, "topic_id"] = cluster_ids

        result.loc[valid_mask, "topic_label"] = [
            ", ".join(self.topic_keywords[cluster_id])
            for cluster_id in cluster_ids
        ]

        return result
