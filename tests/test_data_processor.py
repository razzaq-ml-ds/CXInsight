
import pandas as pd

from src.data_processor import load_and_clean_data


data = {
    "Customer Comments": [
        "The delivery was excellent.",
        "Customer support was very helpful.",
        "The product quality was disappointing."
    ],
    "Rating": [5, 4, 2],
    "Customer ID": [101, 102, 103]
}

df = pd.DataFrame(data)

cleaned_df = load_and_clean_data(
    df,
    feedback_column="customer_comments"
)

print(cleaned_df)

print("\nColumns:")
print(cleaned_df.columns.tolist())

print("\nTotal records:", len(cleaned_df))
