import pandas as pd
from constants import ACCEPTED_RESPONSES, MULTIPLE_ACCEPTED_RESPONSES, BIG_FIVE_COLUMNS


class Survey:
    def __init__(self, data, metadata):
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
    def columns(self):
        return self.data.columns

    @property
    def numeric_columns(self):
        return self.data.select_dtypes(include=["number"]).columns

    def get_question_id(self, col, default_question_id=None):
        if col in self.metadata.columns:
            question_id = self.metadata[col].iloc[0]
            if not pd.isna(question_id):
                return question_id
        return default_question_id

    def get_scoring(self, col):
        if col in self.metadata.columns:
            scoring = self.metadata[col].iloc[1]
            if not pd.isna(scoring):
                return scoring
        return None

    def get_plot_type(self, col, default_plot_type=None):
        if col in self.metadata.columns:
            plot_type = self.metadata[col].iloc[2]
            if not pd.isna(plot_type):
                return plot_type
        return default_plot_type

    def get_title(self, col):
        if col in self.metadata.columns:
            custom_title = self.metadata[col].iloc[3]
            if not pd.isna(custom_title):
                return custom_title
            question_id = self.metadata[col].iloc[0]
            if question_id in BIG_FIVE_COLUMNS:
                return f"I see myself as someone who {col}"
        return col

    def get_column_by_title(self, title):
        for col in self.columns:
            if self.get_title(col) == title:
                return col
        return None

    @classmethod
    def from_file(cls, file_path):
        data = pd.read_csv(file_path)
        data, metadata = cls.preprocess(data)
        return cls(data, metadata)

    @classmethod
    def preprocess(cls, data):
        if data.empty:
            return data, None

        if "Timestamp" in data.columns:
            data.drop(columns=["Timestamp"], inplace=True)

        if "Age" in data.columns:
            data["Age"] = pd.to_numeric(data["Age"], errors="coerce")

        # if a column is empty, drop it
        data.dropna(axis=1, how="all", inplace=True)

        data = data.loc[:, ~data.columns.str.startswith("[OPTIONAL]")]

        metadata = data.iloc[:4]
        data = data.iloc[4:]

        return data, metadata


def concat(surveys, columns):
    metadata = pd.concat([survey.metadata[columns] for survey in surveys])
    data = pd.concat([survey.data[columns] for survey in surveys])
    return Survey(data, metadata)
