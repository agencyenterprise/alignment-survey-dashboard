import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from constants import (
    SCORING_MAPPING,
    BIG_FIVE_CATEGORIES,
    MORAL_FOUNDATIONS_CATEGORIES,
    BIG_FIVE_COLUMNS,
    MORAL_FOUNDATIONS_COLUMNS,
    PREDICTIONS_COLUMNS,
    FLIPPED_DELAY_DISCOUNTING_SCORES,
)
from survey import Survey
from typing import Dict, List, Union


def display_grouped_distribution_plot(st, survey: Survey, category: str) -> None:
    """Displays a grouped distribution plot for a given category within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        category: The category for which the grouped distribution plot is to be displayed.
    """
    category_cols = get_category_columns(survey, category)
    transformed_data = transform_survey_data(survey, category_cols)
    survey.data[category] = transformed_data.mean(axis=1)
    display_standard_plot(
        st,
        survey,
        category,
        {"nbins": 10, "histnorm": "percent"},
    )


def display_individual_vs_community_plot(
    st, survey: Survey, individual_q: str, community_q: str
) -> None:
    """Displays a plot comparing individual versus community responses for given questions.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        individual_q: The column name for individual responses.
        community_q: The column name for community responses.
    """
    plot_type = survey.get_plot_type(individual_q, "histogram")
    individual_data = (
        survey.data[individual_q].str.split(",").explode()
        if plot_type == "histogram-categorized"
        else survey.data[individual_q]
    )
    community_data = (
        survey.data[community_q].str.split(",").explode()
        if plot_type == "histogram-categorized"
        else survey.data[community_q]
    )

    individual_data = individual_data.value_counts(normalize=True).sort_index() * 100
    community_data = community_data.value_counts(normalize=True).sort_index() * 100

    if survey.get_scoring(individual_q) is not None:
        individual_data = individual_data.rename(SCORING_MAPPING)
        community_data = community_data.rename(SCORING_MAPPING)

    all_categories = individual_data.index.union(community_data.index)
    individual_data = individual_data.reindex(all_categories, fill_value=0)
    community_data = community_data.reindex(all_categories, fill_value=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=individual_data.index,
            y=individual_data,
            name="Individual",
            marker_color="blue",
        )
    )
    fig.add_trace(
        go.Bar(
            x=community_data.index,
            y=community_data,
            name="Community",
            marker_color="red",
        )
    )

    title = survey.get_title(individual_q)
    update_layout(
        fig,
        title,
        yaxis_title="percent",
        legend_title=None,
        barmode="group",
        layout_opts={"yaxis": dict(tickformat=",")},
    )

    if survey.get_scoring(individual_q) is not None:
        fig.update_layout(
            xaxis=dict(
                categoryorder="array", categoryarray=list(SCORING_MAPPING.values())
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def display_group_mean_std_graph(st, survey: Survey, group_name: str) -> None:
    """Displays a bar graph of mean scores with standard deviation for a specific group within the survey.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        group_name: The name of the group for which the mean and standard deviation are calculated.
    """
    category_stats = calculate_category_statistics(survey, group_name)
    categories, means, stds = zip(
        *[(k, v["mean"], v["std"]) for k, v in category_stats.items()]
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=means,
                error_y=dict(type="data", array=stds),
                marker_line_color="black",
                marker_line_width=1.5,
                opacity=0.6,
                marker_color=[
                    "#FFD700",
                    "#C0C0C0",
                    "#CD7F32",
                    "#E5E4E2",
                    "#F0E68C",
                    "#ADD8E6",
                ],
            )
        ]
    )
    update_layout(
        fig,
        f"Mean Scores for {group_name} Traits with Standard Deviation",
        "Trait",
        "Mean Score",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_predictions_graph(st, survey: Survey, level_name: str) -> None:
    """Displays a bar graph of predictions for different levels of a specific category within the survey.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        level_name: The name of the level within a specific category for which predictions are displayed.
    """
    column_ids = PREDICTIONS_COLUMNS[level_name]
    columns = {
        col: survey.get_title(col)
        for col in survey.data.columns
        if survey.get_question_id(col, "") in column_ids
    }

    melted_df = survey.data.melt(
        value_vars=columns.keys(), var_name="Question", value_name="Response"
    )
    response_order = [
        "Very unpromising",
        "Somewhat unpromising",
        "Unsure/agnostic",
        "Somewhat promising",
        "Very promising",
    ]
    response_counts = (
        melted_df.groupby(["Question", "Response"]).size().reset_index(name="Counts")
    )
    response_counts["Question"] = response_counts["Question"].map(columns)
    response_counts["Response"] = pd.Categorical(
        response_counts["Response"], categories=response_order, ordered=True
    )

    title = f"Distribution of {level_name} Views on Alignment Approaches"
    fig = px.bar(
        response_counts,
        x="Question",
        y="Counts",
        color="Response",
        title=title,
        barmode="group",
        text="Counts",
        category_orders={"Response": response_order},
    )
    update_layout(fig, title, "Question", "Counts", legend_title="Response Category")
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


def display_standard_plot(
    st, survey: Survey, selected_column: str, plot_kwargs: Dict = {}
) -> None:
    """Displays a standard plot for a selected column within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        selected_column: The column for which the plot is to be displayed.
        plot_kwargs: Additional keyword arguments for the plot.
    """
    plot_type = survey.get_plot_type(selected_column, "histogram")
    fig = None
    if plot_type in ["pie-categorized", "pie"]:
        data = (
            survey.data[selected_column].str.split(",").explode()
            if plot_type == "pie-categorized"
            else survey.data
        )
        fig = px.pie(data, names=selected_column)
        fig.update_traces(hole=0.4, hoverinfo="label+percent+name")
    elif plot_type in ["histogram-categorized", "histogram"]:
        fig_kwargs = {"x": selected_column, "histnorm": "percent"}
        fig_kwargs.update(plot_kwargs)
        if survey.get_scoring(selected_column) is not None:
            survey.data = survey.data.replace(SCORING_MAPPING)
            fig_kwargs["category_orders"] = {
                selected_column: list(SCORING_MAPPING.values())
            }
        data = (
            survey.data[selected_column].str.split(",").explode()
            if plot_type == "histogram-categorized"
            else survey.data
        )

        fig = px.histogram(data, **fig_kwargs)
        fig.update_layout(bargap=0.2)

    if fig:
        update_layout(fig, selected_column, xaxis_title="")
        st.plotly_chart(fig)


def display_correlation_plot(
    st, survey: Survey, x_axis_column: str, y_axis_column: str
) -> None:
    """Displays a scatter plot showing the correlation between two columns within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        x_axis_column: The column to be used as the x-axis.
        y_axis_column: The column to be used as the y-axis.
    """
    grouped_data = (
        survey.data.groupby(y_axis_column)[x_axis_column]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig = px.scatter(
        grouped_data,
        x=y_axis_column,
        y="mean",
        error_y="std",
        labels={
            "mean": "Mean of " + x_axis_column,
            "std": "Std. Dev. of " + x_axis_column,
        },
        title=f"Mean of {x_axis_column} with Std. Dev. vs. {y_axis_column}",
    )

    coeffs = np.polyfit(survey.data[y_axis_column], survey.data[x_axis_column], 1)
    best_fit_line = np.polyval(coeffs, survey.data[y_axis_column])
    fig.add_trace(
        go.Scatter(
            x=survey.data[y_axis_column],
            y=best_fit_line,
            mode="lines",
            name="Best Fit Line",
            line=dict(color="royalblue", dash="dash"),
        )
    )

    update_layout(
        fig,
        f"Mean of {x_axis_column} with Std. Dev. vs. {y_axis_column}",
        y_axis_column,
        "Mean of " + x_axis_column,
    )
    st.plotly_chart(fig)


def display_side_by_side_plot(
    st,
    column_key: str,
    survey: Survey,
    comparison_survey: Survey,
    datasource: str,
    comparison_datasource: str,
) -> None:
    """Displays a side-by-side histogram plot comparing the same column from two different surveys.

    Args:
        st: The Streamlit object used to render the plot.
        column_key: The column key to be compared across surveys.
        survey: The first Survey instance.
        comparison_survey: The second Survey instance to compare against.
        datasource: The name of the first data source.
        comparison_datasource: The name of the second data source.
    """
    column_title = survey.get_title(column_key)
    if survey.get_scoring(column_key) is not None:
        survey.data = survey.data.replace(SCORING_MAPPING)
        comparison_survey.data = comparison_survey.data.replace(SCORING_MAPPING)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=survey.data[column_key],
            name=datasource,
            marker_color="blue",
            opacity=0.75,
            histnorm="percent",
            nbinsx=10,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=comparison_survey.data[column_key],
            name=comparison_datasource,
            marker_color="red",
            opacity=0.75,
            histnorm="percent",
            nbinsx=10,
        )
    )

    update_layout(
        fig,
        column_title,
        yaxis_title="percent",
        legend_title="Dataset",
        barmode="group",
    )
    st.plotly_chart(fig)


def display_side_by_side_grouped_distribution_plot(
    st,
    category: str,
    survey: Survey,
    comparison_survey: Survey,
    datasource: str,
    comparison_datasource: str,
):
    """Displays a side-by-side grouped analysis for a given category.

    Args:
        st: The Streamlit module.
        category: The category to display.
        survey: The first Survey instance.
        comparison_survey: The second Survey instance.
        datasource: The name or source of the first survey data.
        comparison_datasource: The name or source of the second survey data.
    """
    category_cols = get_category_columns(survey, category)

    survey_data_transformed = transform_survey_data(survey, category_cols)
    comparison_data_transformed = transform_survey_data(
        comparison_survey, category_cols
    )
    survey.data[category] = survey_data_transformed.mean(axis=1)
    comparison_survey.data[category] = comparison_data_transformed.mean(axis=1)

    display_side_by_side_plot(
        st,
        category,
        survey,
        comparison_survey,
        datasource,
        comparison_datasource,
    )


def display_delay_discounting_variance(st, survey: Survey) -> None:
    """Displays a histogram plot of the variance in delay discounting responses within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
    """
    dd_columns = [
        col
        for col in survey.data.columns
        if survey.get_question_id(col, "").startswith("delaydiscounting")
    ]
    dd_df = survey.data[dd_columns]

    for col in dd_df.columns:
        dd_df[col] = dd_df[col].map(FLIPPED_DELAY_DISCOUNTING_SCORES).astype(int)
    variance_values = dd_df.var(axis=1)

    fig = px.histogram(
        variance_values,
        nbins=10,
        title="Distribution of Variance in Delay Discounting Responses",
        labels={"value": "Variance"},
        histnorm="percent",
    )
    fig.update_layout(bargap=0.2)

    st.plotly_chart(fig, use_container_width=True)


def update_layout(
    fig: go.Figure,
    title: Union[str, None] = None,
    xaxis_title: Union[str, None] = None,
    yaxis_title: Union[str, None] = None,
    legend_title: Union[str, None] = None,
    barmode: Union[str, None] = None,
    layout_opts: Dict = {},
) -> None:
    """Updates the layout of a Plotly figure with given titles and options.

    Args:
        fig: The Plotly figure object to update.
        title: The title of the plot.
        xaxis_title: The title of the x-axis.
        yaxis_title: The title of the y-axis.
        legend_title: The title of the legend.
        barmode: The bar mode for bar plots (e.g., 'group', 'overlay').
        layout_opts: Additional layout options as a dictionary.
    """
    if title:
        max_length = 100
        if len(title) > max_length:
            words = title.split()
            title_with_breaks = ""
            current_line_length = 0
            for word in words:
                if current_line_length + len(word) <= max_length:
                    title_with_breaks += word + " "
                    current_line_length += len(word) + 1
                else:
                    title_with_breaks += "<br>" + word + " "
                    current_line_length = len(word) + 1

            title = title_with_breaks.rstrip()

    layout_update = {
        "title": {"text": title},
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "legend_title": legend_title,
        "barmode": barmode,
        **layout_opts,
    }
    fig.update_layout(**{k: v for k, v in layout_update.items() if v is not None})


def add_standard_error_trace(
    fig: go.Figure,
    x_data: List[float],
    y_data: List[float],
    error_data: List[float],
    name: str = "Standard Error",
) -> None:
    """Adds a trace for standard error to a Plotly figure.

    Args:
        fig: The Plotly figure object to update.
        x_data: The x data points.
        y_data: The y data points.
        error_data: The standard error values.
        name: The name of the trace.
    """
    fig.add_trace(
        go.Scatter(
            x=x_data, y=y_data, error_y=dict(type="data", array=error_data), name=name
        )
    )


def transform_survey_data(survey: Survey, category_cols: List[str]) -> pd.DataFrame:
    """Transforms survey data based on specified category columns and scoring.

    Args:
        survey: The Survey instance containing the survey data and metadata.
        category_cols: The category columns to transform.

    Returns:
        A pandas DataFrame with transformed data.
    """
    transformed_data = pd.DataFrame()
    for col in category_cols:
        transformed_data[col] = pd.to_numeric(survey.data[col], errors="coerce")
        if survey.get_scoring(col) == "reverse":
            transformed_data[col] = transformed_data[col].apply(lambda x: 6 - x)
    return transformed_data


def get_category_columns(survey: Survey, category: str) -> List[str]:
    """Retrieves the columns corresponding to a specific category within the survey data.

    Args:
        survey: The Survey instance containing the survey data and metadata.
        category: The category for which columns are to be retrieved.

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
        for col in survey.data.columns
        if re.match(rf"{category_key}\d", survey.get_question_id(col, ""))
    ]


def calculate_category_statistics(
    survey: Survey, group_name: str
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """Calculates mean and standard deviation for each category within a specified group in the survey.

    Args:
        survey: The Survey instance containing the survey data and metadata.
        group_name: The name of the group (e.g., "Big Five", "Moral Foundations").

    Returns:
        A dictionary with category names as keys and dictionaries containing 'mean' and 'std' as values.
    """
    category_stats = {}
    for col in survey.data.columns:
        question_id = survey.get_question_id(col, "")
        if question_id not in (BIG_FIVE_COLUMNS + MORAL_FOUNDATIONS_COLUMNS):
            continue

        category = question_id[:-1]
        group_categories = (
            BIG_FIVE_CATEGORIES
            if group_name == "Big Five"
            else MORAL_FOUNDATIONS_CATEGORIES
        )
        if category not in group_categories.keys():
            continue

        category = group_categories[category]
        if category not in category_stats:
            category_stats[category] = {"scores": [], "names": []}
        try:
            numeric_col = pd.to_numeric(survey.data[col], errors="coerce")
            category_stats[category]["scores"].append(numeric_col)
            category_stats[category]["names"].append(numeric_col)
        except Exception as e:
            print(f"Error converting column {col} to numeric: {e}")

    for category, data in category_stats.items():
        if data["scores"]:
            combined_scores = pd.concat(data["scores"], axis=1)
            category_stats[category]["mean"] = combined_scores.mean(axis=1).mean()
            category_stats[category]["std"] = combined_scores.mean(axis=1).std()
        else:
            category_stats[category]["mean"] = None
            category_stats[category]["std"] = None

    return category_stats
