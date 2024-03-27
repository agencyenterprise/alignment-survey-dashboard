import pandas as pd
from typing import Optional, Tuple
from constants import ACCEPTED_RESPONSES, MULTIPLE_ACCEPTED_RESPONSES, BIG_FIVE_COLUMNS


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
    metadata = pd.concat([survey.metadata[columns] for survey in surveys])
    data = pd.concat([survey.data[columns] for survey in surveys])
    return Survey(data, metadata)
