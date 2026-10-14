from functools import lru_cache
from io import BytesIO
import urllib.request
import zipfile

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes


# ============================================================
# DATASET OPTIONS
# ============================================================

DATASET_OPTIONS = {
    "California Housing": (
        "Predict median house value from census-block features."
    ),
    "Diabetes": (
        "Predict a quantitative disease-progression measure "
        "from clinical measurements."
    ),
    "Medical Insurance": (
        "Estimate individual insurance charges from demographic "
        "and lifestyle features."
    ),
    "Advertising": (
        "Predict product sales from advertising spend across media."
    ),
    "Student Performance": (
        "Estimate final student grade from school and "
        "study-related variables."
    ),
}


# ============================================================
# CALIFORNIA HOUSING
# ============================================================

@lru_cache(maxsize=1)
def _california():
    """
    Load the California Housing dataset from scikit-learn.

    Target:
        MedHouseVal
    """

    bunch = fetch_california_housing(as_frame=True)

    df = bunch.frame.copy()

    return (
        df,
        "MedHouseVal",
        (
            "Scikit-learn California Housing dataset. "
            "The target is expressed in units of $100,000. "
            "An initial download may be required."
        ),
    )


# ============================================================
# DIABETES
# ============================================================

@lru_cache(maxsize=1)
def _diabetes():
    """
    Load the Diabetes dataset from scikit-learn.

    Target:
        target
    """

    bunch = load_diabetes(as_frame=True)

    df = bunch.frame.copy()

    return (
        df,
        "target",
        (
            "Scikit-learn Diabetes dataset. The target is a "
            "quantitative measure of disease progression "
            "one year after baseline."
        ),
    )


# ============================================================
# MEDICAL INSURANCE
# ============================================================

@lru_cache(maxsize=1)
def _insurance():
    """
    Load the Medical Insurance dataset.

    Target:
        charges
    """

    urls = [
        (
            "https://raw.githubusercontent.com/"
            "stedy/Machine-Learning-with-R-datasets/"
            "master/insurance.csv"
        ),
        (
            "https://raw.githubusercontent.com/"
            "JovianML/opendatasets/master/"
            "data/medical-insurance/insurance.csv"
        ),
    ]

    last_error = None

    for url in urls:
        try:
            # Download the CSV with an explicit timeout.
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=20,
            ) as response:
                csv_data = response.read()

            # Read the downloaded CSV.
            df = pd.read_csv(BytesIO(csv_data))

            # Standardize column names.
            df.columns = [
                str(column).strip().lower()
                for column in df.columns
            ]

            # Validate the target column.
            required_columns = {
                "age",
                "sex",
                "bmi",
                "children",
                "smoker",
                "region",
                "charges",
            }

            missing_columns = required_columns - set(df.columns)

            if missing_columns:
                raise ValueError(
                    "Medical Insurance dataset is missing "
                    f"columns: {sorted(missing_columns)}"
                )

            if df.empty:
                raise ValueError(
                    "Medical Insurance dataset is empty."
                )

            # Convert numerical columns safely.
            numeric_columns = [
                "age",
                "bmi",
                "children",
                "charges",
            ]

            for column in numeric_columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            # Remove invalid records.
            df = df.dropna(
                subset=list(required_columns)
            ).reset_index(drop=True)

            if df.empty:
                raise ValueError(
                    "No valid records remain after cleaning."
                )

            return (
                df,
                "charges",
                (
                    "Medical Insurance dataset containing "
                    "demographic and lifestyle features. "
                    "The target is individual insurance charges. "
                    "Categorical variables are one-hot encoded "
                    "automatically by the preprocessing pipeline."
                ),
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to download the Medical Insurance CSV. "
        "Check your internet connection or try again later. "
        f"Last error: {last_error}"
    )

# ============================================================
# ADVERTISING
# ============================================================

@lru_cache(maxsize=1)
def _advertising():
    """
    Load the Advertising dataset.

    Features:
        TV
        Radio
        Newspaper

    Target:
        Sales

    Expected shape:
        200 rows and 4 columns, excluding the index column.
    """

    urls = [
        # Official source associated with the ISL textbook.
        "https://www.statlearning.com/s/Advertising.csv",

        # Fallback mirrors.
        (
            "https://raw.githubusercontent.com/"
            "nguyen-toan/ISLR/master/dataset/Advertising.csv"
        ),
        (
            "https://raw.githubusercontent.com/"
            "selva86/datasets/master/Advertising.csv"
        ),
        (
            "https://raw.githubusercontent.com/"
            "justmarkham/DAT8/master/data/Advertising.csv"
        ),
    ]

    last_error = None

    for url in urls:
        try:
            # Download with a timeout instead of relying on
            # pandas to open a remote URL directly.
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=20,
            ) as response:
                content = response.read()

            # Parse the downloaded CSV.
            df = pd.read_csv(BytesIO(content))

            # Normalize column names.
            df.columns = [
                str(column).strip().lower()
                for column in df.columns
            ]

            # Remove unnecessary index columns.
            df = df.drop(
                columns=[
                    "unnamed: 0",
                    "x",
                    "index",
                ],
                errors="ignore",
            )

            # Standardize the expected column names.
            df = df.rename(
                columns={
                    "tv": "TV",
                    "radio": "Radio",
                    "newspaper": "Newspaper",
                    "sales": "Sales",
                }
            )

            # Validate the downloaded dataset.
            required_columns = [
                "TV",
                "Radio",
                "Newspaper",
                "Sales",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Advertising dataset is missing columns: "
                    f"{missing_columns}. "
                    f"Received columns: {df.columns.tolist()}"
                )

            # Keep only the columns needed for regression.
            df = df[required_columns].copy()

            # Ensure every column is numeric.
            for column in required_columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

            # Remove rows containing invalid values.
            df = df.dropna().reset_index(drop=True)

            if df.empty:
                raise ValueError(
                    "The Advertising dataset contains no "
                    "valid observations."
                )

            if len(df) < 100:
                raise ValueError(
                    "The downloaded Advertising dataset has "
                    f"only {len(df)} rows; expected approximately "
                    "200 observations."
                )

            # Ensure all values are finite.
            numeric_values = df[required_columns].to_numpy()

            if not pd.notna(numeric_values).all():
                raise ValueError(
                    "The Advertising dataset contains invalid "
                    "numeric values."
                )

            return (
                df,
                "Sales",
                (
                    "Advertising dataset containing TV, Radio, "
                    "and Newspaper advertising expenditure. "
                    "The target is Sales. The original dataset "
                    "contains 200 market observations."
                ),
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to download the Advertising CSV from the "
        "available sources. Check your internet connection "
        "or try again later. "
        f"Last error: {last_error}"
    )


# ============================================================
# STUDENT PERFORMANCE
# ============================================================

@lru_cache(maxsize=1)
def _student():
    """
    Load the UCI Student Performance Mathematics dataset.

    Target:
        G3

    G1 and G2 are retained as input features.
    """

    url = (
        "https://archive.ics.uci.edu/"
        "ml/machine-learning-databases/00320/student.zip"
    )

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            payload = response.read()

        with zipfile.ZipFile(BytesIO(payload)) as zf:

            # Find the Mathematics dataset in the archive.
            matching_files = [
                filename
                for filename in zf.namelist()
                if filename.endswith("student-mat.csv")
            ]

            if not matching_files:
                raise FileNotFoundError(
                    "student-mat.csv was not found "
                    "inside the downloaded ZIP archive."
                )

            with zf.open(matching_files[0]) as file:
                df = pd.read_csv(
                    file,
                    sep=";",
                )

        if "G3" not in df.columns:
            raise ValueError(
                "The Student Performance dataset "
                "does not contain the G3 target column."
            )

        if df.empty:
            raise ValueError(
                "The Student Performance dataset is empty."
            )

        return (
            df,
            "G3",
            (
                "UCI Student Performance Mathematics dataset. "
                "Categorical features are one-hot encoded "
                "automatically by the preprocessing pipeline. "
                "G1 and G2 are retained as input features, so "
                "results reflect those inputs."
            ),
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to download the UCI Student Performance "
            f"dataset: {exc}"
        ) from exc


# ============================================================
# MAIN DATASET LOADER
# ============================================================

def load_dataset(name):
    """
    Load the selected dataset.

    Returns:
        df       : pandas DataFrame
        target   : target column name
        info     : dataset description
    """

    dataset_loaders = {
        "California Housing": _california,
        "Diabetes": _diabetes,
        "Medical Insurance": _insurance,
        "Advertising": _advertising,
        "Student Performance": _student,
    }

    if name not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(dataset_loaders.keys())}"
        )

    return dataset_loaders[name]()