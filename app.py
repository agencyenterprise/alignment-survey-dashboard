from enum import Enum
from filter_dataframe import filter_dataframe
from survey import Survey, concat as concat_surveys
import plots
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from constants import (
    QUESTION_PAIRS,
    BIG_FIVE_CATEGORIES,
    MORAL_FOUNDATIONS_CATEGORIES,
)


class DatasetType(Enum):
    ALIGNMENT = "Alignment Researchers"
    EA = "EA"
    COMBINED = "Combined"
    SIDE_BY_SIDE = "Side by Side"


class AnalysesType(Enum):
    RAW = "Raw Distribution"
    GROUPED = "Grouped Distribution"
    INDIVIDUAL_VS_GROUP = "Individual vs. Group"
    CORRELATION = "Correlation"
    MISCELLANEOUS = "Miscellaneous"


MISC_GRAPH_TYPES = {
    "Mean Scores for Big Five Traits with Std. Dev.": plots.display_group_mean_std_graph,
    "Mean Scores for Moral Foundations Traits with Std. Dev.": plots.display_group_mean_std_graph,
    "Distribution of Individual Views on Alignment Approaches": plots.display_predictions_graph,
    "Distribution of Community Views on Alignment Approaches": plots.display_predictions_graph,
    "Distribution of Variance in Delay Discounting Responses": plots.display_delay_discounting_variance,
}


def select_all_callback(key_suffix, options):
    def callback():
        st.session_state[f"graph_select_{key_suffix}"] = options

    return callback


def select_graphs(st, key_suffix, options, title="Select Graph(s):"):
    if st.button("Select All", key=f"select_all_{key_suffix}"):
        select_all_callback(key_suffix, options)()

    selected_graphs = st.multiselect(
        title, options=options, key=f"graph_select_{key_suffix}"
    )

    return selected_graphs


def get_graphs_for_analysis(analysis_type, survey: Survey):
    graphs = []

    if analysis_type == AnalysesType.GROUPED.value:
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

    if analysis_type == AnalysesType.INDIVIDUAL_VS_GROUP.value:
        for individual_q, community_q in QUESTION_PAIRS.items():
            if individual_q in survey.columns:
                graphs.append(survey.get_title(individual_q) + " vs. Community View")
    if analysis_type == AnalysesType.MISCELLANEOUS.value:
        graphs.extend(MISC_GRAPH_TYPES.keys())

    if analysis_type == AnalysesType.RAW.value:
        return [survey.get_title(col) for col in survey.columns]

    return graphs


def display_side_by_side_analysis(
    survey,
    datasource,
    comparison_survey,
    comparison_datasource,
    st,
    key_suffix="",
):
    survey.data = filter_dataframe(survey.data, scope=datasource, key_suffix=key_suffix)
    comparison_survey.data = filter_dataframe(
        comparison_survey.data,
        scope=comparison_datasource,
        key_suffix=f"comparison_{key_suffix}",
    )

    common_columns = list(
        set(survey.columns).intersection(set(comparison_survey.columns))
    )
    column_titles = [survey.get_title(col) for col in common_columns]

    selected_columns = select_graphs(
        st, f"comparison_select_{key_suffix}", column_titles
    )

    for selected_title in selected_columns:
        selected_column_key = next(
            col for col in common_columns if survey.get_title(col) == selected_title
        )

        plots.display_side_by_side_plot(
            st,
            selected_column_key,
            survey,
            comparison_survey,
            datasource,
            comparison_datasource,
        )


def display_standard_analysis(survey, datasource, st=st, key_suffix=""):
    survey.data = filter_dataframe(survey.data, scope=datasource, key_suffix=key_suffix)

    all_graphs = [state.value for state in AnalysesType]
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
    graph_types,
    survey,
    st,
    key_suffix,
):
    if graph_types == AnalysesType.CORRELATION.value:
        display_correlation_analysis(st, survey, key_suffix)
    elif graph_types == AnalysesType.RAW.value:
        display_raw_analysis(
            st,
            survey,
            key_suffix,
        )
    elif graph_types in [
        AnalysesType.GROUPED.value,
        AnalysesType.INDIVIDUAL_VS_GROUP.value,
    ]:
        display_grouped_analysis(st, graph_types, survey, key_suffix)
    elif graph_types == AnalysesType.MISCELLANEOUS.value:
        display_miscellaneous_analysis(st, survey, key_suffix)


def display_correlation_analysis(st, survey, key_suffix):
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


def display_raw_analysis(st, survey: Survey, key_suffix):
    direct_columns = get_graphs_for_analysis(AnalysesType.RAW.value, survey)

    selected_columns = select_graphs(st, key_suffix, direct_columns)

    for selected_column in selected_columns:
        selected_column_key = survey.get_column_by_title(selected_column)

        display_selected_plot(
            st,
            survey,
            selected_column_key,
        )


def display_grouped_analysis(st, graph_types, survey, key_suffix):
    selected_columns = select_graphs(
        st, key_suffix, get_graphs_for_analysis(graph_types, survey)
    )

    for selected_column in selected_columns:
        display_selected_plot(
            st,
            survey,
            selected_column,
        )


def display_miscellaneous_analysis(st, survey, key_suffix):
    selected_columns = select_graphs(
        st,
        key_suffix,
        get_graphs_for_analysis(AnalysesType.MISCELLANEOUS.value, survey),
    )

    for selected_analysis in selected_columns:
        display_selected_plot(
            st,
            survey,
            selected_analysis,
        )


def display_selected_plot(st, survey, selected_column):
    if selected_column in MISC_GRAPH_TYPES.keys():
        if selected_column.endswith(" Traits with Std. Dev."):
            group_name = selected_column.split("for ")[1].split(" Traits")[0]
            plots.display_group_mean_std_graph(st, survey, group_name)
        elif selected_column.endswith(" Views on Alignment Approaches"):
            level_name = selected_column.split(" of ")[1].split(" Views")[0]
            plots.display_predictions_graph(st, survey, level_name)
        elif selected_column.endswith(" in Delay Discounting Responses"):
            plots.display_delay_discounting_variance(st, survey)
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
        plots.display_standard_plot(st, survey, selected_column)


def initialize_surveys():
    """Initialize surveys and return them as a dictionary."""
    alignment_survey = Survey.from_file("alignment_data.csv")
    ea_survey = Survey.from_file("ea_data.csv")
    common_columns = list(
        set(alignment_survey.columns).intersection(set(ea_survey.columns))
    )
    combined_survey = concat_surveys([alignment_survey, ea_survey], common_columns)
    return {"Alignment": alignment_survey, "EA": ea_survey, "Combined": combined_survey}


def get_survey(surveys, choice):
    """Return the survey based on the given choice."""
    return surveys.get(choice)


def generate_response(survey: Survey, input_query):
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.2,
    )
    agent = create_pandas_dataframe_agent(
        llm, survey.data, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    )
    response = agent.run(input_query)
    return response


def handle_gpt4_query(dataframe, st):
    query_placeholder = st.empty()
    query_text = query_placeholder.text_input(
        "Enter a query to be answered by GPT-4:",
        value=st.session_state.get("query_text", ""),
    )
    if query_text:
        with st.spinner("Running query..."):
            response = generate_response(dataframe, query_text)
        st.write("GPT-4:", response)


def main():
    st.set_page_config(page_title="Data Analysis Dashboard")
    analysis_state = st.radio("Dataset", [state.value for state in DatasetType])

    surveys = initialize_surveys()

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
        survey = get_survey(surveys, survey_name)
        display_standard_analysis(survey, analysis_state, st=st, key_suffix="")

    elif analysis_state == DatasetType.SIDE_BY_SIDE.value:
        choices = ["Alignment", "EA", "Combined"]
        side1_choice = st.selectbox(
            "Choose data for Side 1:", choices, key="side1", index=0
        )
        side2_choice = st.selectbox(
            "Choose data for Side 2:", choices, key="side2", index=1
        )
        survey1, survey2 = get_survey(surveys, side1_choice), get_survey(
            surveys, side2_choice
        )
        display_side_by_side_analysis(
            survey1, side1_choice, survey2, side2_choice, st, key_suffix="side1"
        )

    else:
        st.error("Invalid analysis state selected.")

    if "survey" in locals() and survey is not None:
        handle_gpt4_query(survey, st)


if __name__ == "__main__":
    main()
