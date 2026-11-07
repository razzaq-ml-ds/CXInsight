
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


def normalize_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df


def read_raw_data(source):

    try:
        if hasattr(source, "seek"):
            source.seek(0)

        df = pd.read_csv(source, encoding="utf-8-sig")

    except Exception as error:
        raise ValueError(f"Unable to read CSV file: {error}")

    df = normalize_column_names(df)

    return df


def load_and_clean_data(source, feedback_column=None):

    # Accept either an uploaded CSV or an already-read DataFrame
    if isinstance(source, pd.DataFrame):
        df = normalize_column_names(source)
    else:
        df = read_raw_data(source)

    # Map the user-selected column to our standard column name
    if feedback_column is not None:

        selected_column = (
            str(feedback_column)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if selected_column not in df.columns:
            raise ValueError(
                f"Selected feedback column '{feedback_column}' "
                "does not exist in the dataset."
            )

        df.rename(
            columns={selected_column: "feedback_text"},
            inplace=True
        )

    else:
        # Backward compatibility for previously supported datasets
        column_aliases = {
            "review_text": "feedback_text"
        }

        for old_name, new_name in column_aliases.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(
                    columns={old_name: new_name},
                    inplace=True
                )

    # Map optional columns
    column_aliases = {
        "division_name": "category",
        "department_name": "category"
    }

    for old_name, new_name in column_aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(
                columns={old_name: new_name},
                inplace=True
            )

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only supported columns
    available_columns = [
        column
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if column in df.columns
    ]

    df = df[available_columns].copy()

    # Clean feedback text
    df["feedback_text"] = (
        df["feedback_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[df["feedback_text"] != ""].copy()

    # Clean ratings
    if "rating" in df.columns:

        df["rating"] = pd.to_numeric(
            df["rating"],
            errors="coerce"
        )

        df.loc[
            ~df["rating"].between(1, 5),
            "rating"
        ] = None

    # Clean NPS scores
    if "nps_score" in df.columns:

        df["nps_score"] = pd.to_numeric(
            df["nps_score"],
            errors="coerce"
        )

        df.loc[
            ~df["nps_score"].between(0, 10),
            "nps_score"
        ] = None

    # Clean dates
    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    df.reset_index(drop=True, inplace=True)

    return df
