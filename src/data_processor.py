
import pandas as pd


REQUIRED_COLUMNS = ["feedback_text"]

OPTIONAL_COLUMNS = [
    "feedback_id",
    "customer_id",
    "rating",
    "date",
    "category",
    "nps_score"
]


def load_and_clean_data(source):
    """
    Load customer feedback from a CSV file
    and perform basic data cleaning.
    """

    # Step 1: Read the CSV file
    try:
        df = pd.read_csv(source, encoding="utf-8-sig")
    except Exception as error:
        raise ValueError(f"Unable to read CSV file: {error}")

    # Step 2: Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Step 3: Validate required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Step 4: Keep only supported columns
    available_columns = [
        column
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if column in df.columns
    ]

    df = df[available_columns].copy()

    # Step 5: Clean feedback text
    df["feedback_text"] = (
        df["feedback_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Step 6: Remove empty feedback
    df = df[
        df["feedback_text"] != ""
    ].copy()

    # Step 7: Validate rating values (1-5)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        df.loc[
            ~df["rating"].between(1, 5),
            "rating"
        ] = None

    # Step 8: Validate NPS scores (0-10)
    if "nps_score" in df.columns:
        df["nps_score"] = pd.to_numeric(
            df["nps_score"],
            errors="coerce"
        )

        df.loc[
            ~df["nps_score"].between(0, 10),
            "nps_score"
        ] = None

    # Step 9: Convert dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    # Step 10: Reset index
    df.reset_index(drop=True, inplace=True)

    return df
