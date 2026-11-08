
from src.data_processor import load_and_clean_data
from src.sentiment_analyzer import SentimentAnalyzer
from src.analytics import CustomerAnalytics


# Load and clean data
df = load_and_clean_data("data/sample_feedback.csv")

# Perform sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
df = sentiment_analyzer.analyze_dataframe(df)

# Initialize analytics
analytics = CustomerAnalytics(df)

# Display business metrics
print("\nCUSTOMER EXPERIENCE ANALYTICS")
print("-" * 35)

print("Total Feedback:", analytics.get_total_feedback())

print("\nSentiment Distribution:")
print(analytics.get_sentiment_distribution())

print("\nSentiment Percentages:")
print(analytics.get_sentiment_percentages())

print("\nAverage Rating:", analytics.get_average_rating())

print("Net Promoter Score:", analytics.get_nps())

# Verify results
assert analytics.get_total_feedback() == 30
assert analytics.get_average_rating() is not None
assert analytics.get_nps() is not None

print("\nAll analytics tests passed!")
