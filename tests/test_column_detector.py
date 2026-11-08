
import pandas as pd
from src.column_detector import ColumnDetector


# TEST 1: Clear customer feedback column
data = {
    "Customer_ID": [101, 102, 103],
    "Review Text": [
        "The product quality is excellent and delivery was fast.",
        "I am disappointed with the material and overall quality.",
        "The customer support team was very helpful."
    ],
    "Rating": [5, 2, 4]
}

df = pd.DataFrame(data)

detector = ColumnDetector()

best_column, confidence, candidates = detector.detect_feedback_column(df)

print("\n========== TEST 1 ==========")
print("Detected column:", best_column)
print("Confidence:", confidence)


# TEST 2: Generic text column
data = {
    "ID": [101, 102, 103],
    "text": [
        "The delivery was late and the packaging was damaged.",
        "I really enjoyed the shopping experience.",
        "Customer support resolved my issue quickly."
    ],
    "score": [2, 5, 4]
}

df = pd.DataFrame(data)

best_column, confidence, candidates = detector.detect_feedback_column(df)

print("\n========== TEST 2 ==========")
print("Detected column:", best_column)
print("Confidence:", confidence)


# TEST 3: Non-feedback text
data = {
    "text": ["Delhi", "Mumbai", "Agra"],
    "customer_id": [101, 102, 103],
    "rating": [5, 4, 3]
}

df = pd.DataFrame(data)

best_column, confidence, candidates = detector.detect_feedback_column(df)

print("\n========== TEST 3 ==========")
print("Detected column:", best_column)
print("Confidence:", confidence)
