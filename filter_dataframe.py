# Originally copied from https://github.com/tylerjrichards/st-filter-dataframe/blob/main/streamlit_app.py
# Adapted for multiple filters support
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def filter_dataframe(df: pd.DataFrame, scope: str, key_suffix: str) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns with the ability to add multiple filters.

    Args:
        df (pd.DataFrame): Original dataframe
        key_suffix (str): Suffix for unique session state keys

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox(f"Add filters for {scope}", key=f"modify_{key_suffix}")

    if not modify:
        return df

    df = df.copy()

    filter_count_key = f"filter_count_{key_suffix}"
    if filter_count_key not in st.session_state:
        st.session_state[filter_count_key] = 1

    def add_filter():
        st.session_state[filter_count_key] += 1

    st.button(
        "Add another filter", on_click=add_filter, key=f"modify_other_{key_suffix}"
    )

    for i in range(st.session_state[filter_count_key]):
        with st.container():
            st.write(f"Filter {i + 1}")
            selected_column = st.selectbox(
                "Filter dataframe on", df.columns, key=f"{key_suffix}_filter_col_{i}"
            )
            df = apply_filter_to_column(df, selected_column, key_suffix, i)

    return df


def apply_filter_to_column(df, column, key_suffix, filter_id):
    """
    Apply user-defined filter to a column in the dataframe and return the filtered dataframe.

    Args:
        df (pd.DataFrame): Dataframe to filter
        column (str): Column name to apply filter on
        key_suffix (str): Suffix for unique session state keys
        filter_id (int): Identifier for the filter, used for session state keys

    Returns:
        pd.DataFrame: The filtered dataframe
    """
    left, right = st.columns((1, 20))
    left.write("↳")
    filtered_df = df

    if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
        user_cat_input = right.multiselect(
            f"Values for {column}",
            df[column].unique(),
            default=list(df[column].unique()),
            key=f"{key_suffix}_{filter_id}_cat_input",
        )
        filtered_df = filtered_df[filtered_df[column].isin(user_cat_input)]
    elif is_numeric_dtype(df[column]):
        _min = float(df[column].min())
        _max = float(df[column].max())
        step = (_max - _min) / 100
        user_num_input = right.slider(
            f"Values for {column}",
            _min,
            _max,
            (_min, _max),
            step=step,
            key=f"{key_suffix}_{filter_id}_num_input",
        )
        filtered_df = filtered_df[filtered_df[column].between(*user_num_input)]
    elif is_datetime64_any_dtype(df[column]):
        user_date_input = right.date_input(
            f"Values for {column}",
            value=(df[column].min(), df[column].max()),
            key=f"{key_suffix}_{filter_id}_date_input",
        )
        if len(user_date_input) == 2:
            user_date_input = tuple(map(pd.to_datetime, user_date_input))
            start_date, end_date = user_date_input
            filtered_df = filtered_df.loc[
                filtered_df[column].between(start_date, end_date)
            ]
    else:
        user_text_input = right.text_input(
            f"Substring or regex in {column}",
            key=f"{key_suffix}_{filter_id}_text_input",
        )
        if user_text_input:
            filtered_df = filtered_df[
                filtered_df[column].str.contains(user_text_input, na=False)
            ]

    return filtered_df
