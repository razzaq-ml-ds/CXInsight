
from src.data_processor import load_and_clean_data
from src.topic_analyzer import TopicAnalyzer


df = load_and_clean_data("data/sample_feedback.csv")

analyzer = TopicAnalyzer(n_clusters=5)


result = analyzer.analyze_dataframe(df)

print("\nFeedback and Topic Assignments:")

print(
    result[
        ["feedback_text", "topic_id", "topic_label"]
    ].to_string(index=False)
)


print("\nDiscovered Topics:")

for topic_id, keywords in analyzer.topic_keywords.items():
    print(f"Topic {topic_id}: {', '.join(keywords)}")

assert len(result) == 30
assert result["topic_id"].notna().all()
assert result["topic_label"].notna().all()
assert len(analyzer.topic_keywords) == 5

print("\nAll topic analysis tests passed!")
