from src.data_processor import load_and_clean_data
from src.sentiment_analyzer import SentimentAnalyzer


# Load our sample dataset
df = load_and_clean_data("data/sample_feedback.csv")

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

# Analyze all customer feedback
result = analyzer.analyze_dataframe(df)

# Display results
print(
    result[
        [
            "feedback_text",
            "compound_score",
            "sentiment"
        ]
    ].to_string(index=False)
)

# Display sentiment distribution
print("\nSentiment Distribution:")

print(
    result["sentiment"].value_counts()
)

# Verify output
assert len(result) == 30
assert "compound_score" in result.columns
assert "sentiment" in result.columns

print("\nAll tests passed!")

# Test individual feedback
print("\nIndividual sentiment tests:")

print(analyzer.analyze_text(
    "This product is absolutely amazing!"
))

print(analyzer.analyze_text(
    "Terrible service. I am extremely disappointed."
))

print(analyzer.analyze_text(
    "The product arrived yesterday."
))

print(analyzer.analyze_text(""))