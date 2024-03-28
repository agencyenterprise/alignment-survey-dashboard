from enum import Enum
from filter_dataframe import filter_dataframe
from survey import Survey, concat as concat_surveys
import plots
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Callable
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from constants import (
    QUESTION_PAIRS,
    BIG_FIVE_CATEGORIES,
    MORAL_FOUNDATIONS_CATEGORIES,
)


class DatasetType(Enum):
    """Enum to represent the type of dataset being analyzed."""

    ALIGNMENT = "Alignment Researchers"
    EA = "EA"
    COMBINED = "Combined"
    SIDE_BY_SIDE = "Side by Side"


class AnalysisType(Enum):
    """Enum to represent the type of dataset being analyzed."""

    RAW = "Raw Distribution"
    GROUPED = "Grouped Distribution"
    INDIVIDUAL_VS_GROUP = "Individual vs. Group"
    CORRELATION = "Correlation"
    MISCELLANEOUS = "Miscellaneous"


MISC_GRAPH_TYPES: Dict[str, Callable] = {
    "Mean Scores for Big Five Traits with Std. Dev.": plots.display_group_mean_std_graph,
    "Mean Scores for Moral Foundations Traits with Std. Dev.": plots.display_group_mean_std_graph,
    "Distribution of Individual Views on Alignment Approaches": plots.display_predictions_graph,
    "Distribution of Community Views on Alignment Approaches": plots.display_predictions_graph,
    "Distribution of Variance in Delay Discounting Responses": plots.display_delay_discounting_variance,
    "Distribution of k Values (lower = less future discounting)": plots.display_delay_discounting_k_values,
}


def select_all_callback(key_suffix: str, options: List[str]) -> Callable:
    """Returns a callback function that selects all options in a multi-select widget.

    Args:
        key_suffix: The suffix to be used for the session state key.
        options: The options to be selected.

    Returns:
        A callback function.
    """

    def callback():
        st.session_state[f"graph_select_{key_suffix}"] = options

    return callback


def select_graphs(
    st, key_suffix: str, options: List[str], title: str = "Select Graph(s):"
) -> List[str]:
    """Renders a "Select All" button and a multi-select widget for selecting graphs.

    Args:
        st: The Streamlit module.
        key_suffix: The suffix to be used for the session state key.
        options: The options to be displayed in the multi-select widget.
        title: The title of the multi-select widget.

    Returns:
        A list of selected options.
    """

    def select_all_callback():
        """The callback is needed so that the button can be placed below the multi-select widget."""
        st.session_state[f"graph_select_{key_suffix}"] = options

    selected_graphs = st.multiselect(
        title, options=options, key=f"graph_select_{key_suffix}"
    )

    st.button(
        "Select All", key=f"select_all_{key_suffix}", on_click=select_all_callback
    )

    return selected_graphs


def get_graphs_for_analysis(analysis_type: str, survey: Survey) -> List[str]:
    """Determines the appropriate graphs to display based on the analysis type.

    Args:
        analysis_type: The type of analysis being performed.
        survey: The Survey instance containing survey data and metadata.

    Returns:
        A list of graph titles or descriptions.
    """
    graphs = []
    if analysis_type == AnalysisType.GROUPED.value:
        graphs.extend(
            [
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
        )

    if analysis_type == AnalysisType.INDIVIDUAL_VS_GROUP.value:
        for individual_q, community_q in QUESTION_PAIRS.items():
            if individual_q in survey.columns:
                graphs.append(survey.get_title(individual_q) + " vs. Community View")
    if analysis_type == AnalysisType.MISCELLANEOUS.value:
        graphs.extend(MISC_GRAPH_TYPES.keys())

    if analysis_type == AnalysisType.RAW.value:
        return [survey.get_title(col) for col in survey.columns]

    return graphs


def display_side_by_side_analysis(
    survey: Survey,
    datasource: str,
    comparison_survey: Survey,
    comparison_datasource: str,
    st,
    key_suffix: str = "",
) -> None:
    """Displays a side-by-side analysis comparing two surveys, including grouped analysis.

    Args:
        survey: The first Survey instance.
        datasource: The name or source of the first survey data.
        comparison_survey: The second Survey instance to compare against.
        comparison_datasource: The name or source of the second survey data.
        st: The Streamlit module.
        key_suffix: A suffix to differentiate session state keys if necessary.
    """
    survey.data = filter_dataframe(survey.data, scope=datasource, key_suffix=key_suffix)
    comparison_survey.data = filter_dataframe(
        comparison_survey.data,
        scope=comparison_datasource,
        key_suffix=f"comparison_{key_suffix}",
    )

    analysis_types = [AnalysisType.RAW.value, AnalysisType.GROUPED.value]
    selected_analysis_type = st.selectbox(
        "Select Analysis Type:",
        options=analysis_types,
        index=0,
        key=f"analysis_type_{key_suffix}",
    )

    if selected_analysis_type == AnalysisType.GROUPED.value:
        graphs = get_graphs_for_grouped_analysis()
    else:
        common_columns = list(
            set(survey.columns).intersection(set(comparison_survey.columns))
        )
        graphs = [survey.get_title(col) for col in common_columns]

    selected_graphs = select_graphs(st, f"comparison_select_{key_suffix}", graphs)

    for graph in selected_graphs:
        if selected_analysis_type == AnalysisType.GROUPED.value:
            category = graph.split("for ")[1]
            plots.display_side_by_side_grouped_distribution_plot(
                st,
                category,
                survey,
                comparison_survey,
                datasource,
                comparison_datasource,
            )
        else:
            selected_column_key = next(
                col for col in common_columns if survey.get_title(col) == graph
            )

            plots.display_side_by_side_plot(
                st,
                selected_column_key,
                survey,
                comparison_survey,
                datasource,
                comparison_datasource,
            )


def get_graphs_for_grouped_analysis() -> List[str]:
    """Get graphs suitable for grouped analysis from two surveys.

    Args:
        survey: The first Survey instance.
        comparison_survey: The second Survey instance.

    Returns:
        A list of graph titles suitable for grouped analysis.
    """
    return [
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


def display_standard_analysis(
    survey: Survey, datasource: str, st=st, key_suffix: str = ""
) -> None:
    """Displays the standard analysis for a given survey.

    Args:
        survey: The Survey instance to analyze.
        datasource: The name or source of the survey data.
        st: The Streamlit module.
        key_suffix: A suffix to differentiate session state keys if necessary
    """
    survey.data = filter_dataframe(survey.data, scope=datasource, key_suffix=key_suffix)

    all_graphs = [state.value for state in AnalysisType]
    graph_types = st.selectbox(
        "Analysis Type:", all_graphs, key=f"graph_type_{key_suffix}"
    )
    handle_analysis_selection(
        graph_types,
        survey,
        st,
        key_suffix,
    )


def handle_analysis_selection(
    graph_types: str, survey: Survey, st, key_suffix: str
) -> None:
    """Handles the selection of analysis type and displays the corresponding analysis.

    Args:
        graph_types: The selected type of analysis to perform.
        survey: The Survey instance containing the data for analysis.
        st: The Streamlit module.
        key_suffix: A suffix to differentiate session state keys if necessary.
    """
    if graph_types == AnalysisType.CORRELATION.value:
        display_correlation_analysis(st, survey, key_suffix)
    elif graph_types == AnalysisType.RAW.value:
        display_raw_analysis(
            st,
            survey,
            key_suffix,
        )
    elif graph_types in [
        AnalysisType.GROUPED.value,
        AnalysisType.INDIVIDUAL_VS_GROUP.value,
    ]:
        display_grouped_analysis(st, graph_types, survey, key_suffix)
    elif graph_types == AnalysisType.MISCELLANEOUS.value:
        display_miscellaneous_analysis(st, survey, key_suffix)


def display_correlation_analysis(st, survey: Survey, key_suffix: str) -> None:
    """Displays a correlation analysis between two selected columns from the survey.

    Args:
        st: The Streamlit module.
        survey: The Survey instance containing the data for analysis.
        key_suffix: A suffix to differentiate session state keys if necessary.
    """
    col1, col2 = st.columns(2)
    with col1:
        x_axis_column = st.selectbox(
            "X Axis:", options=survey.numeric_columns, key=f"x_axis_select_{key_suffix}"
        )
    with col2:
        y_axis_options = [col for col in survey.numeric_columns if col != x_axis_column]
        y_axis_column = st.selectbox(
            "Y Axis:", options=y_axis_options, key=f"y_axis_select_{key_suffix}"
        )
    plots.display_correlation_plot(st, survey, x_axis_column, y_axis_column)


def display_raw_analysis(st, survey: Survey, key_suffix: str) -> None:
    """Displays raw analysis for selected columns from the survey.

    Args:
        st: The Streamlit module.
        survey: The Survey instance containing the data for analysis.
        key_suffix: A suffix to differentiate session state keys if necessary.
    """
    direct_columns = get_graphs_for_analysis(AnalysisType.RAW.value, survey)

    selected_columns = select_graphs(st, key_suffix, direct_columns)

    for selected_column in selected_columns:
        selected_column_key = survey.get_column_by_title(selected_column)

        display_selected_plot(
            st,
            survey,
            selected_column_key,
        )


def display_grouped_analysis(
    st, graph_types: str, survey: Survey, key_suffix: str
) -> None:
    """Displays grouped analysis for selected categories within the survey.

    Args:
        st: The Streamlit module.
        graph_types: The selected type of analysis to perform.
        survey: The Survey instance containing the data for analysis.
        key_suffix: A suffix to differentiate session state keys if necessary.
    """
    selected_columns = select_graphs(
        st, key_suffix, get_graphs_for_analysis(graph_types, survey)
    )

    for selected_column in selected_columns:
        display_selected_plot(
            st,
            survey,
            selected_column,
        )


def display_miscellaneous_analysis(st, survey: Survey, key_suffix: str) -> None:
    """Displays miscellaneous analyses based on the selected analysis type.

    Args:
        st: The Streamlit module.
        survey: The Survey instance containing the data for analysis.
        key_suffix: A suffix to differentiate session state keys if necessary.
    """
    selected_columns = select_graphs(
        st,
        key_suffix,
        get_graphs_for_analysis(AnalysisType.MISCELLANEOUS.value, survey),
    )

    for selected_analysis in selected_columns:
        display_selected_plot(
            st,
            survey,
            selected_analysis,
        )


def display_selected_plot(st, survey: Survey, selected_column: str) -> None:
    """Displays the plot for a selected column or analysis type.

    Args:
        st: The Streamlit module.
        survey: The Survey instance containing the data for analysis.
        selected_column: The column or analysis type selected for plotting.
    """
    if selected_column in MISC_GRAPH_TYPES.keys():
        if selected_column.endswith(" Traits with Std. Dev."):
            group_name = selected_column.split("for ")[1].split(" Traits")[0]
            plots.display_group_mean_std_graph(st, survey, group_name)
        elif selected_column.endswith(" Views on Alignment Approaches"):
            level_name = selected_column.split(" of ")[1].split(" Views")[0]
            plots.display_predictions_graph(st, survey, level_name)
        elif selected_column.endswith(" in Delay Discounting Responses"):
            plots.display_delay_discounting_variance(st, survey)
        elif selected_column.endswith(" k Values (lower = less future discounting)"):
            plots.display_delay_discounting_k_values(st, survey)
    elif selected_column.endswith(" vs. Community View"):
        base_title = selected_column.replace(" vs. Community View", "")
        individual_q = next(
            (q for q in survey.data.columns if survey.get_title(q) == base_title),
            None,
        )
        community_q = QUESTION_PAIRS.get(individual_q)
        plots.display_individual_vs_community_plot(
            st, survey, individual_q, community_q
        )
    elif selected_column.startswith("Overall Distribution for"):
        category = selected_column.split("for ")[1]
        plots.display_grouped_distribution_plot(st, survey, category)
    else:
        plots.display_raw_distribution_plot(st, survey, selected_column)


def get_survey(choice: str) -> Optional[Survey]:
    """Returns a new Survey instance based on the user's choice, created on the fly.

    Args:
        choice: The user's choice of survey.

    Returns:
        A new Survey instance based on the choice, or None if the choice is invalid.
    """
    if choice == "Alignment":
        return Survey.from_file("alignment_data.csv")
    elif choice == "EA":
        return Survey.from_file("ea_data.csv")
    elif choice == "Combined":
        alignment_survey = Survey.from_file("alignment_data.csv")
        ea_survey = Survey.from_file("ea_data.csv")
        common_columns = list(
            set(alignment_survey.columns).intersection(set(ea_survey.columns))
        )
        combined_survey = concat_surveys([alignment_survey, ea_survey], common_columns)
        return combined_survey
    else:
        return None


def generate_response(dataframes: List[pd.DataFrame], input_query: str) -> str:
    """Generates a response using GPT-4 based on the input query and dataframes.

    Args:
        dataframes: A list of pandas DataFrames containing the data for GPT-4 to analyze.
        input_query: The user's input query for GPT-4 to process.

    Returns:
        The response generated by GPT-4.
    """
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.2,
    )
    agent = create_pandas_dataframe_agent(
        llm, dataframes, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    )
    response = agent.run(input_query)
    return response


def handle_gpt4_query(dataframes: List[pd.DataFrame], st) -> None:
    """Handles user queries and displays responses from GPT-4 for each provided DataFrame.

    Args:
        dataframes: A list of pandas DataFrames containing the data for GPT-4 to analyze.
        st: The Streamlit module.
    """
    query_placeholder = st.empty()
    query_text = query_placeholder.text_input(
        "Enter a query to be answered by GPT-4:",
        value=st.session_state.get("query_text", ""),
    )
    if query_text:
        with st.spinner("Running query..."):
            response = generate_response(dataframes, query_text)
        st.write("GPT-4:", response)


def main() -> None:
    """The main function to run the Streamlit application."""
    st.set_page_config(page_title="Data Analysis Dashboard")
    analysis_state = st.radio("Dataset", [state.value for state in DatasetType])

    if analysis_state in [
        DatasetType.ALIGNMENT.value,
        DatasetType.EA.value,
        DatasetType.COMBINED.value,
    ]:
        survey_name = (
            "Alignment"
            if analysis_state == DatasetType.ALIGNMENT.value
            else "EA" if analysis_state == DatasetType.EA.value else "Combined"
        )
        survey = get_survey(survey_name)
        display_standard_analysis(survey, analysis_state, st=st, key_suffix="")

    elif analysis_state == DatasetType.SIDE_BY_SIDE.value:
        choices = ["Alignment", "EA", "Combined"]
        side1_choice = st.selectbox(
            "Choose data for Side 1:", choices, key="side1", index=0
        )
        side2_choice = st.selectbox(
            "Choose data for Side 2:", choices, key="side2", index=1
        )
        survey1, survey2 = get_survey(side1_choice), get_survey(side2_choice)
        display_side_by_side_analysis(
            survey1, side1_choice, survey2, side2_choice, st, key_suffix="side1"
        )
        handle_gpt4_query([survey1.data, survey2.data], st)

    else:
        st.error("Invalid analysis state selected.")

    if "survey" in locals() and survey is not None:
        handle_gpt4_query(survey, st)


if __name__ == "__main__":
    main()
