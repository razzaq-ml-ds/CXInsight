
import pandas as pd


class ColumnDetector:

    def __init__(self):
        self.feedback_keywords = [
            "feedback",
            "review",
            "comment",
            "text",
            "message",
            "response",
            "remark",
            "verbatim"
        ]

    def detect_feedback_column(self, df):

        candidates = []

        for column in df.columns:

            values = df[column].dropna().astype(str).str.strip()

            if values.empty:
                continue

            column_name = column.lower().replace("_", " ")

            name_score = 0

            if any(
                keyword in column_name
                for keyword in self.feedback_keywords
            ):
                name_score = 50

            non_empty_values = values[values != ""]

            if non_empty_values.empty:
                continue

            average_length = non_empty_values.str.len().mean()

            sentence_ratio = (
                non_empty_values.str.contains(r"\s", regex=True).mean()
            )

            content_score = 0

            if average_length >= 20:
                content_score += 30

            if sentence_ratio >= 0.5:
                content_score += 20

            total_score = name_score + content_score

            candidates.append({
                "column": column,
                "name_score": name_score,
                "content_score": content_score,
                "total_score": total_score
            })

        candidates.sort(
            key=lambda item: item["total_score"],
            reverse=True
        )

        if not candidates:
            return None, "Low", candidates

        best_candidate = candidates[0]

        if best_candidate["total_score"] >= 80:
            confidence = "High"
            best_column = best_candidate["column"]

        elif best_candidate["total_score"] >= 50:
            confidence = "Medium"
            best_column = None

        else:
            confidence = "Low"
            best_column = None

        return best_column, confidence, candidates
