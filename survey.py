import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple, List
from constants import (
    ACCEPTED_RESPONSES,
    MULTIPLE_ACCEPTED_RESPONSES,
    BIG_FIVE_COLUMNS,
    BIG_FIVE_CATEGORIES,
    MORAL_FOUNDATIONS_CATEGORIES,
    PREDICTIONS_MAPPING,
    FLIPPED_DELAY_DISCOUNTING_SCORES,
    K_VALUES,
)


class Survey:
    """Represents a survey, holding both the data and metadata associated with it.

    Attributes:
        metadata (pd.DataFrame): Metadata for the survey, providing additional context like question IDs and scoring information.
        data (pd.DataFrame): The actual survey data, including responses.

    Methods:
        columns: Returns the columns of the survey data.
        numeric_columns: Returns the columns of the survey data that contain numeric values.
        get_question_id: Retrieves the question ID for a given column, if available.
        get_scoring: Retrieves the scoring information for a given column, if available.
        get_plot_type: Retrieves the plot type for a given column, if available.
        get_title: Constructs a title for a given column based on metadata or default formatting.
        get_column_by_title: Finds a column that matches a given title.
        from_file: Constructs a Survey instance from a CSV file.
        preprocess: Preprocesses raw data into a format suitable for Survey.
    """

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame) -> None:
        """Initializes the Survey with data and metadata, applying necessary transformations."""
        self.metadata = metadata
        self.data = data

        for col in data.columns:
            if self.get_scoring(col) is not None:
                data[col] = pd.to_numeric(data[col], errors="coerce")

            if col in ACCEPTED_RESPONSES.keys():
                data[col] = data[col].apply(
                    lambda x: x if x in ACCEPTED_RESPONSES[col] else "Other"
                )

            if col in MULTIPLE_ACCEPTED_RESPONSES.keys():
                split_responses = data[col].str.split(",").explode()
                cleaned_responses = split_responses.str.strip()
                categorized_responses = cleaned_responses.apply(
                    lambda x: x if x in MULTIPLE_ACCEPTED_RESPONSES[col] else "Other"
                )

                data[col] = categorized_responses.groupby(
                    categorized_responses.index
                ).agg(lambda x: ",".join(x))

    @property
    def columns(self) -> pd.Index:
        """Returns the columns of the survey data."""
        return self.data.columns

    @property
    def numeric_columns(self) -> pd.Index:
        """Returns the columns of the survey data that contain numeric values."""
        return self.data.select_dtypes(include=["number"]).columns

    def get_question_id(
        self, col: str, default_question_id: Optional[str] = None
    ) -> Optional[str]:
        """Retrieves the question ID for a given column, if available."""
        if col in self.metadata.columns:
            question_id = self.metadata[col].iloc[0]
            if not pd.isna(question_id):
                return question_id
        return default_question_id

    def get_scoring(self, col: str) -> Optional[str]:
        """Retrieves the scoring information for a given column, if available."""
        if col in self.metadata.columns:
            scoring = self.metadata[col].iloc[1]
            if not pd.isna(scoring):
                return scoring
        return None

    def get_plot_type(
        self, col: str, default_plot_type: Optional[str] = None
    ) -> Optional[str]:
        """Retrieves the plot type for a given column, if available."""
        if col in self.metadata.columns:
            plot_type = self.metadata[col].iloc[2]
            if not pd.isna(plot_type):
                return plot_type
        return default_plot_type

    def get_title(self, col: str) -> str:
        """Constructs a title for a given column based on metadata or default formatting."""
        if col in self.metadata.columns:
            custom_title = self.metadata[col].iloc[3]
            if not pd.isna(custom_title):
                return custom_title
            question_id = self.metadata[col].iloc[0]
            if question_id in BIG_FIVE_COLUMNS:
                return f"I see myself as someone who {col}"
        return col

    def get_column_by_title(self, title: str) -> Optional[str]:
        """Finds a column that matches a given title."""
        for col in self.columns:
            if self.get_title(col) == title:
                return col
        return None

    def get_prediction_columns(self, level_name=None) -> List[str]:
        """Returns the columns for questions about predictions.

        Args:
            level_name (str): The level of the survey data to filter by.
            Can be "Individual", "Community", or "All".

        Returns:
            List[str]: A list of columns for questions about predictions.
        """
        if level_name == "Individual":
            category_keys = ["iep", "iap"]
        elif level_name == "Community":
            category_keys = ["cep", "cap"]
        else:
            category_keys = ["iep", "iap", "cep", "cap"]

        return [
            col
            for col in self.data.columns
            if any(
                re.match(rf"{category_key}\d", self.get_question_id(col, ""))
                for category_key in category_keys
            )
        ]

    def get_category_columns(self, category: Optional[str] = None) -> List[str]:
        """Retrieves the columns corresponding to a specific category within the survey data.

        Args:
            category: The category for which columns are to be retrieved.
            Can be one of the keys in BIG_FIVE_CATEGORIES or MORAL_FOUNDATIONS_CATEGORIES.

        Returns:
            A list of column names corresponding to the specified category.
        """
        category_key = next(
            (
                k
                for k, v in (BIG_FIVE_CATEGORIES | MORAL_FOUNDATIONS_CATEGORIES).items()
                if v == category
            ),
            None,
        )
        return [
            col
            for col in self.data.columns
            if re.match(rf"{category_key}\d", self.get_question_id(col, ""))
        ]

    def get_delay_discounting_columns(self) -> List[str]:
        """Retrieves the columns corresponding to delay discounting questions within the survey data.

        Returns:
            A list of column names corresponding to delay discounting questions.
        """
        return [
            col
            for col in self.data.columns
            if self.get_question_id(col, "").startswith("delaydiscounting")
        ]

    def get_category_data_distribution(self, category_cols: List[str]) -> pd.Series:
        """Converts category columns to numeric values and applies reverse scoring where necessary.
        Then calculates the mean of the transformed data across the specified category columns.

        Args:
            category_cols: The category columns to transform.

        Returns:
            A Series containing the mean of the transformed data across the specified category columns.
        """
        transformed_data = pd.DataFrame()
        for col in category_cols:
            transformed_data[col] = pd.to_numeric(self.data[col], errors="coerce")
            if self.get_scoring(col) == "reverse":
                transformed_data[col] = transformed_data[col].apply(lambda x: 6 - x)
        return transformed_data.mean(axis=1)

    def get_numeric_data(self) -> pd.DataFrame:
        """Returns the numeric data from the survey.
        Converts the categorical data to numeric data using the PREDICTIONS_MAPPING.
        Adds group distribution columns for each category in the survey.
        Calculates k-values for delay discounting questions.

        Args:
            survey: The Survey instance containing the data for analysis.

        Returns:
            The numeric data from the survey.
        """
        numeric_data = self.data.copy()
        numeric_data = numeric_data[self.numeric_columns]

        prediction_columns = self.get_prediction_columns()
        for col in prediction_columns:
            numeric_data[col] = self.data[col].map(
                {v: k for k, v in PREDICTIONS_MAPPING.items()}
            )

        group_distribution_columns = [
            f"Overall Distribution for {category_name}"
            for category_name in (
                set(
                    [
                        category_name
                        for category_name in (
                            BIG_FIVE_CATEGORIES | MORAL_FOUNDATIONS_CATEGORIES
                        ).values()
                    ]
                )
            )
        ]
        for col in group_distribution_columns:
            category = col.split("for ")[1]
            category_cols = self.get_category_columns(category)
            numeric_data[col] = self.get_category_data_distribution(category_cols)

        numeric_data["k-value"] = self.get_k_values()

        return numeric_data

    def get_k_values(self):
        def calculate_k(row, k_values_dict):
            """Calculates the k value for a given row of delay discounting responses."""
            ks = [k_values_dict[col]["1"] for col in row.index if row[col] == 1]
            return (
                np.mean(ks) if ks else np.nan
            )  # Return NaN if there are no choices for delayed reward

        dd_columns = self.get_delay_discounting_columns()
        dd_df = self.data[dd_columns]
        dd_df.columns = [self.get_question_id(col, "") for col in dd_df.columns]

        for col in dd_df.columns:
            dd_df[col] = dd_df[col].map(FLIPPED_DELAY_DISCOUNTING_SCORES).astype(int)

        return dd_df.apply(lambda row: calculate_k(row, K_VALUES), axis=1)

    @classmethod
    def from_file(cls, file_path: str) -> "Survey":
        """Constructs a Survey instance from a CSV file."""
        data = pd.read_csv(file_path)
        data, metadata = cls.preprocess(data)
        return cls(data, metadata)

    @classmethod
    def preprocess(cls, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocesses raw data into a format suitable for Survey.

        This includes dropping unnecessary columns, converting columns to numeric where applicable, and separating metadata.

        Args:
            data (pd.DataFrame): The raw survey data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessed data and metadata.
        """
        if data.empty:
            return data, None

        if "Timestamp" in data.columns:
            data.drop(columns=["Timestamp"], inplace=True)

        if "Age" in data.columns:
            data["Age"] = pd.to_numeric(data["Age"], errors="coerce")

        # Drop columns that are completely empty
        data.dropna(axis=1, how="all", inplace=True)

        # Remove optional columns
        data = data.loc[:, ~data.columns.str.startswith("[OPTIONAL]")]

        # Assume the first 4 rows are metadata
        metadata = data.iloc[:4]
        data = data.iloc[4:]

        return data, metadata


def concat(surveys: list[Survey], columns: list[str]) -> Survey:
    """Concatenates multiple Survey instances into a single Survey.

    This function combines the data and metadata from multiple surveys, filtering by specified columns.

    Args:
        surveys (list[Survey]): A list of Survey instances to concatenate.
        columns (list[str]): The columns to include in the concatenated Survey.

    Returns:
        Survey: A new Survey instance containing the concatenated data and metadata.
    """
    metadata = pd.concat([survey.metadata for survey in surveys], ignore_index=True)
    data = pd.concat([survey.data for survey in surveys], ignore_index=True)

    original_order = surveys[0].data.columns
    filtered_columns = [col for col in original_order if col in columns]
    data = data[filtered_columns]
    metadata = metadata.loc[:, filtered_columns]

    return Survey(data, metadata)
