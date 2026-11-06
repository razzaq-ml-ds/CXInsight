
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

    try:
        df = pd.read_csv(source, encoding="utf-8-sig")
    except Exception as error:
        raise ValueError(f"Unable to read CSV file: {error}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )


    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    available_columns = [
        column
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if column in df.columns
    ]

    df = df[available_columns].copy()

    df["feedback_text"] = (
        df["feedback_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    df = df[
        df["feedback_text"] != ""
    ].copy()


    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        df.loc[
            ~df["rating"].between(1, 5),
            "rating"
        ] = None


    if "nps_score" in df.columns:
        df["nps_score"] = pd.to_numeric(
            df["nps_score"],
            errors="coerce"
        )

        df.loc[
            ~df["nps_score"].between(0, 10),
            "nps_score"
        ] = None


    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )


    df.reset_index(drop=True, inplace=True)

    return df
