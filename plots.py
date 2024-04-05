import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, mannwhitneyu
import scipy.stats as stats
from constants import (
    SCORING_MAPPING,
    PREDICTIONS_MAPPING,
    BIG_FIVE_CATEGORIES,
    MORAL_FOUNDATIONS_CATEGORIES,
    BIG_FIVE_COLUMNS,
    MORAL_FOUNDATIONS_COLUMNS,
    FLIPPED_DELAY_DISCOUNTING_SCORES,
    PREDICTIONS,
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
    category_cols = survey.get_category_columns(category)
    plot_single(
        st,
        survey.get_category_data_distribution(category_cols),
        category,
        "histogram",
        {"nbins": 10, "histnorm": "percent"},
    )


def display_individual_vs_community_plot(
    st,
    survey: Survey,
    individual_q: str,
    community_q: str,
    show_descriptive_stats: bool,
) -> None:
    """Displays a plot comparing individual versus community responses for given questions.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        individual_q: The column name for individual responses.
        community_q: The column name for community responses.
        show_descriptive_stats: Whether to display descriptive statistics for the data.
    """
    fig = plot_side_by_side(
        st,
        survey.get_title(individual_q),
        format_survey_data_for_plotting(survey, individual_q),
        "Ground truth",
        format_survey_data_for_plotting(survey, community_q),
        "Predictions",
        return_fig=True,
    )

    individual_data = format_survey_data_for_statistics(survey, individual_q)
    community_data = format_survey_data_for_statistics(survey, community_q)

    if (
        show_descriptive_stats
        and pd.to_numeric(individual_data, errors="coerce").notnull().all()
    ):
        survey_data = individual_data.dropna()
        comparison_data = community_data.dropna()

        mean1, std1 = survey_data.mean(), survey_data.std()
        mean2, std2 = comparison_data.mean(), comparison_data.std()

        u_stat, p_value = mannwhitneyu(
            survey_data, comparison_data, alternative="two-sided"
        )

        stats_text = (
            f"Ground truth Mean: {mean1:.2f}, Std. Dev.: {std1:.2f}<br>"
            f"Predictions Mean: {mean2:.2f}, Std. Dev.: {std2:.2f}<br>"
            f"Mann-Whitney U: {u_stat}, p-value: {p_value:.4f}"
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.55,
            showarrow=False,
            text=stats_text,
            align="left",
            font=dict(size=12),
            borderwidth=1,
            borderpad=4,
        )

        fig.update_layout(margin=dict(b=120))
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
    column_names = survey.get_prediction_columns(level_name)
    columns = {
        col: PREDICTIONS[survey.get_question_id(col)[1:]] for col in column_names
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

    title = f"Distribution of {level_name} Views"
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


def format_survey_data_for_plotting(survey, selected_column) -> pd.Series:
    """Formats survey data for plotting based on the selected column.

    Args:
        survey: The survey data to be formatted.
        selected_column: The column for which the data is to be formatted.


    Returns:
        The formatted data for the plot.
    """
    data = survey.data[selected_column]
    scoring = survey.get_scoring(selected_column)
    is_prediction = selected_column in survey.get_prediction_columns()
    plot_type = survey.get_plot_type(selected_column, "histogram")
    is_comma_separated = plot_type in ["histogram-categorized", "pie-categorized"]

    if scoring is not None:
        data = data.replace(SCORING_MAPPING)
        categories = [
            "Strongly disagree",
            "Somewhat disagree",
            "Neutral/agnostic",
            "Somewhat agree",
            "Strongly agree",
        ]
        data = pd.Series(pd.Categorical(data, categories=categories, ordered=True))
        data = data.sort_values()

    if is_prediction:
        categories = [
            "Very unpromising",
            "Somewhat unpromising",
            "Unsure/agnostic",
            "Somewhat promising",
            "Very promising",
        ]
        data = pd.Series(pd.Categorical(data, categories=categories, ordered=True))
        data = data.sort_values()

    if is_comma_separated:
        data = data.str.split(",").explode().str.strip()

    return data


def format_survey_data_for_statistics(survey, selected_column) -> pd.Series:
    """Converts some survey data to numeric for statistical analysis.

    Args:
        survey: The survey data to be formatted.
        selected_column: The column for which the data is to be formatted.


    Returns:
        The formatted data for the plot.
    """
    data = survey.data[selected_column]
    is_prediction = selected_column in survey.get_prediction_columns()
    plot_type = survey.get_plot_type(selected_column, "histogram")
    is_comma_separated = plot_type in ["histogram-categorized", "pie-categorized"]

    if is_prediction:
        data = data.replace({v: k for k, v in PREDICTIONS_MAPPING.items()})

    return data


def display_raw_distribution_plot(st, survey: Survey, selected_column: str) -> None:
    """Displays a standard plot for a selected column within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        selected_column: The column for which the plot is to be displayed.
        plot_kwargs: Additional keyword arguments for the plot.
    """
    plot_type = survey.get_plot_type(selected_column, "histogram")
    plot_single(
        st,
        format_survey_data_for_plotting(survey, selected_column),
        (
            survey.get_title(selected_column)
            if plot_type in ["histogram", "histogram-categorized"]
            else selected_column
        ),
        plot_type,
        zoom_to_fit_categories=True,
    )


def plot_single(
    st,
    data: pd.Series,
    series_name: str,
    plot_type: str = "histogram",
    plot_kwargs: dict = {},
    zoom_to_fit_categories: bool = False,
    return_fig: bool = False,
) -> None:
    """Displays a standard plot for a selected column within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        data: The data to be plotted.
        series_name: The name of the series to be displayed.
        plot_type: The type of plot to display (e.g., 'histogram', 'pie').
        plot_kwargs: Additional keyword arguments for the plot including options for highlighting.
        zoom_to_fit_categories: Whether to zoom to fit categories for the plot in order to make empty categories visible.
        return_fig: Whether to return the Plotly figure object.
    """
    fig = None

    if plot_type in ["histogram-categorized", "histogram"]:
        highlight_value = plot_kwargs.pop("highlight_value", None)
        if highlight_value is not None and isinstance(highlight_value, (int, float)):
            highlight_value = f"{highlight_value:.2f}"
        highlight_color = plot_kwargs.pop("highlight_color", "red")
        fig_kwargs = {"histnorm": "percent"}
        fig_kwargs.update(plot_kwargs)

        if isinstance(data.dtype, pd.CategoricalDtype):
            data_counts = data.value_counts().reindex(data.cat.categories, fill_value=0)
            x_values = data_counts.index.tolist()
            y_values = data_counts.values

            fig = px.histogram(x=x_values, y=y_values, **fig_kwargs)

            if highlight_value is not None and highlight_value in x_values:
                fig.add_annotation(
                    x=highlight_value,
                    y=0,
                    text=f"You",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-0,
                    bgcolor=highlight_color,
                    font=dict(color="white"),
                )
            if zoom_to_fit_categories:  # Make empty categories visible as well
                num_categories = len(x_values)
                fig.update_xaxes(range=[-0.5, num_categories - 0.5])
        else:
            fig = px.histogram(data, **fig_kwargs)
            if highlight_value is not None:
                fig.add_annotation(
                    x=highlight_value,
                    y=0,
                    text=f"You: {highlight_value}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=0,
                    bgcolor=highlight_color,
                    font=dict(color="white"),
                )
        fig.update_layout(bargap=0.2, showlegend=False)
    elif plot_type in ["pie-categorized", "pie"]:
        fig = px.pie(data, names=series_name, **plot_kwargs)
        fig.update_traces(hole=0.4, hoverinfo="label+percent+name")

    if fig:
        update_layout(fig, series_name, xaxis_title="", yaxis_title="percent")
        if return_fig:
            return fig
        else:
            st.plotly_chart(fig)


def display_correlation_plot(
    st, data: pd.DataFrame, x_axis_column, y_axis_column
) -> None:
    """Displays a scatter plot showing the correlation between two columns within a dataframe.

    Args:
        st: The Streamlit object used to render the plot.
        data: The data frame containing the columns to be plotted.
        x_axis_column: The column to be used as the x-axis.
        y_axis_column: The column to be used as the y-axis.
    """
    grouped_data = (
        data.groupby(x_axis_column)[y_axis_column].agg(["mean", "std"]).reset_index()
    )
    fig = px.scatter(
        grouped_data,
        x=x_axis_column,
        y="mean",
        error_y="std",
        labels={
            "mean": "Mean of " + y_axis_column,
            "std": "Std. Dev. of " + y_axis_column,
        },
        title=f"Mean of {y_axis_column} with Std. Dev. vs. {x_axis_column}",
    )

    coeffs = np.polyfit(data[x_axis_column], data[y_axis_column], 1)
    best_fit_line = np.polyval(coeffs, data[x_axis_column])

    fig.add_trace(
        go.Scatter(
            x=data[x_axis_column],
            y=best_fit_line,
            mode="lines",
            name="Best Fit Line",
            line=dict(color="royalblue", dash="dash"),
        )
    )

    corr, p_value = pearsonr(data[x_axis_column], data[y_axis_column])
    fig.update_xaxes(title_text=f"{x_axis_column} (r={corr:.2f}, p={p_value:.3g})")

    st.plotly_chart(fig)


def split_text(text, max_length=95):
    """Splits text into multiple lines if longer than max_length, trying to break at spaces."""
    if len(text) <= max_length:
        return text
    else:
        words = text.split(" ")
        split_text = ""
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += " " + word if current_line else word
            else:
                split_text += current_line + "<br>"
                current_line = word
        split_text += current_line
        return split_text


def display_correlation_matrix(st, dataframe: pd.DataFrame) -> None:
    """Displays a correlation matrix with p-values for the given data frame, handling NaNs and infs.

    Args:
        st: The Streamlit module.
        dataframe: The data frame for which the correlation matrix is to be displayed.
    """
    cleaned_df = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    corr_matrix = np.zeros((cleaned_df.shape[1], cleaned_df.shape[1]))
    p_values_matrix = np.zeros((cleaned_df.shape[1], cleaned_df.shape[1]))

    for i, column1 in enumerate(cleaned_df.columns):
        for j, column2 in enumerate(cleaned_df.columns):
            if i <= j:
                corr, p_value = stats.pearsonr(cleaned_df[column1], cleaned_df[column2])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetric matrix
                p_values_matrix[i, j] = p_value
                p_values_matrix[j, i] = p_value  # Symmetric matrix
            else:
                # Upper triangle already computed
                continue

    corr_df = pd.DataFrame(
        corr_matrix, index=cleaned_df.columns, columns=cleaned_df.columns
    )
    p_values_df = pd.DataFrame(
        p_values_matrix, index=cleaned_df.columns, columns=cleaned_df.columns
    )

    hovertext = list()
    for yi, yy in enumerate(corr_df.index):
        hovertext.append(list())
        for xi, xx in enumerate(corr_df.columns):
            formatted_xx = split_text(xx)  # Split x if too long
            formatted_yy = split_text(yy)  # Split y if too long
            hovertext[-1].append(
                f"x: {formatted_xx}<br>y: {formatted_yy}<br>r: {corr_df.iloc[yi, xi]:.2f}<br>p-value: {p_values_df.iloc[yi, xi]:.4f}"
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.to_numpy(),
            x=corr_df.columns,
            y=corr_df.index,
            hoverongaps=False,
            colorscale="Viridis",
            hoverinfo="text",
            text=hovertext,
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.update_layout(title_text="Correlation Matrix", title_x=0.0)
    st.plotly_chart(fig, use_container_width=True)


def format_applied_filters(datasource, applied_filters, max_words=20, max_length=50):
    concatenated_filters = ", ".join(
        applied_filter["column"] for applied_filter in applied_filters
    )
    if len(concatenated_filters) > max_length:
        concatenated_filters = concatenated_filters[:max_length] + "..."

    formatted_filters = split_text(concatenated_filters, max_words)
    return f"{datasource} ({formatted_filters})" if formatted_filters else datasource


def display_side_by_side_plot(
    st,
    column_key: str,
    survey: Survey,
    comparison_survey: Survey,
    datasource: str,
    comparison_datasource: str,
    show_descriptive_stats: bool = False,
) -> None:
    """Displays a side-by-side histogram plot comparing the same column from two different surveys.

    Args:
        st: The Streamlit object used to render the plot.
        column_key: The column key to be compared across surveys.
        survey: The first Survey instance.
        comparison_survey: The second Survey instance to compare against.
        datasource: The name of the first data source.
        comparison_datasource: The name of the second data source.
        show_descriptive_stats: Whether to display descriptive statistics for the data.
    """
    column_title = survey.get_title(column_key)
    fig = plot_side_by_side(
        st,
        column_title,
        format_survey_data_for_plotting(survey, column_key),
        format_applied_filters(datasource, survey.applied_filters),
        format_survey_data_for_plotting(comparison_survey, column_key),
        format_applied_filters(
            comparison_datasource, comparison_survey.applied_filters
        ),
        return_fig=True,
    )

    survey_data = format_survey_data_for_statistics(survey, column_key).dropna()
    comparison_data = format_survey_data_for_statistics(
        comparison_survey, column_key
    ).dropna()

    if (
        show_descriptive_stats
        and pd.to_numeric(survey_data, errors="coerce").notnull().all()
    ):
        mean1, std1 = survey_data.mean(), survey_data.std()
        mean2, std2 = comparison_data.mean(), comparison_data.std()

        u_stat, p_value = mannwhitneyu(
            survey_data, comparison_data, alternative="two-sided"
        )

        stats_text = (
            f"{format_applied_filters(datasource, survey.applied_filters, max_words=95)} Mean: {mean1:.2f}, Std. Dev.: {std1:.2f}<br>"
            f"{format_applied_filters(comparison_datasource, comparison_survey.applied_filters, max_words=95)} Mean: {mean2:.2f}, Std. Dev.: {std2:.2f}<br>"
            f"Mann-Whitney U: {u_stat}, p-value: {p_value:.4f}"
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.5,
            showarrow=False,
            text=stats_text,
            align="left",
            font=dict(size=12),
            borderwidth=1,
            borderpad=4,
        )

        fig.update_layout(margin=dict(b=120))

    st.plotly_chart(fig, use_container_width=True)


def plot_side_by_side(
    st,
    title: str,
    data: pd.Series,
    datasource: str,
    comparison_data: pd.Series,
    comparison_datasource: str,
    nbins: int = 0,
    return_fig: bool = False,
):
    """Plots two histograms side by side for comparison.

    Args:
        st: The Streamlit object used to render the plot.
        title: The title of the plot.
        data: The first series to plot.
        datasource: The name of the first series.
        comparison_data: The second series to plot.
        comparison_datasource: The name of the second series.
        nbins: The number of bins to use for the histogram, only applies if data is not categorical.
        return_fig: Whether to return the Plotly figure object.
    """
    fig = go.Figure()

    for plot_data, plot_source, plot_color in zip(
        [data, comparison_data],
        [datasource, comparison_datasource],
        ["blue", "red"],
    ):
        if isinstance(plot_data.dtype, pd.CategoricalDtype):
            data_counts = plot_data.value_counts().reindex(
                plot_data.cat.categories, fill_value=0
            )
            data_percentages = data_counts / data_counts.sum() * 100
            fig.add_trace(
                go.Bar(
                    x=plot_data.cat.categories,
                    y=data_percentages,
                    name=plot_source,
                    marker_color=plot_color,
                    opacity=0.75,
                )
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=plot_data,
                    name=plot_source,
                    marker_color=plot_color,
                    opacity=0.75,
                    histnorm="percent",
                    nbinsx=nbins,
                )
            )

    update_layout(
        fig,
        title=title,
        yaxis_title="percent",
        legend_title="Dataset",
        barmode="group",
    )
    if return_fig:
        return fig
    else:
        st.plotly_chart(fig, use_container_width=True)


def display_side_by_side_grouped_distribution_plot(
    st,
    category: str,
    survey: Survey,
    comparison_survey: Survey,
    datasource: str,
    comparison_datasource: str,
    show_descriptive_stats: bool = False,
):
    """Displays a side-by-side grouped analysis for a given category.

    Args:
        st: The Streamlit module.
        category: The category to display.
        survey: The first Survey instance.
        comparison_survey: The second Survey instance.
        datasource: The name or source of the first survey data.
        comparison_datasource: The name or source of the second survey data.
        show_descriptive_stats: Whether to display descriptive statistics for the data.
    """
    category_cols = survey.get_category_columns(category)
    survey_grouped_data = survey.get_category_data_distribution(category_cols)
    comparison_grouped_data = comparison_survey.get_category_data_distribution(
        category_cols
    )

    fig = plot_side_by_side(
        st,
        category,
        survey_grouped_data,
        format_applied_filters(datasource, survey.applied_filters),
        comparison_grouped_data,
        format_applied_filters(
            comparison_datasource, comparison_survey.applied_filters
        ),
        nbins=10,
        return_fig=True,
    )

    if show_descriptive_stats:
        mean1, std1 = survey_grouped_data.mean(), survey_grouped_data.std()
        mean2, std2 = comparison_grouped_data.mean(), comparison_grouped_data.std()

        u_stat, p_value = mannwhitneyu(
            survey_grouped_data, comparison_grouped_data, alternative="two-sided"
        )

        stats_text = (
            f"{format_applied_filters(datasource, survey.applied_filters, max_words=95)} Mean: {mean1:.2f}, Std. Dev.: {std1:.2f}<br>"
            f"{format_applied_filters(comparison_datasource, comparison_survey.applied_filters, max_words=95)} Mean: {mean2:.2f}, Std. Dev.: {std2:.2f}<br>"
            f"Mann-Whitney U: {u_stat}, p-value: {p_value:.4f}"
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.5,
            showarrow=False,
            text=stats_text,
            align="left",
            font=dict(size=12),
            borderwidth=1,
            borderpad=4,
        )

        fig.update_layout(margin=dict(b=120))

    st.plotly_chart(fig, use_container_width=True)


def display_delay_discounting_variance(
    st,
    survey: Survey,
    comparison_survey: Survey = None,
    datasource=None,
    comparison_datasource=None,
    show_descriptive_stats=False,
) -> None:
    """Displays a histogram plot of the variance in delay discounting responses within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        comparison_survey: An optional second Survey instance for comparison.
        datasource: The name of the first data source.
        comparison_datasource: The name of the second data source.
        show_descriptive_stats: Whether to display descriptive statistics for the data.
    """

    def get_variance_values(df, dd_columns):
        """Calculates the variance of delay discounting responses for each participant."""
        dd_df = df[dd_columns]

        for col in dd_df.columns:
            dd_df[col] = dd_df[col].map(FLIPPED_DELAY_DISCOUNTING_SCORES).astype(int)
        return dd_df.var(axis=1)

    dd_columns = [
        col
        for col in survey.data.columns
        if survey.get_question_id(col, "").startswith("delaydiscounting")
    ]

    dd_variance = get_variance_values(survey.data, dd_columns)
    if comparison_survey:
        comparison_dd_variance = get_variance_values(comparison_survey.data, dd_columns)
        fig = plot_side_by_side(
            st,
            "Distribution of Variance in Delay Discounting",
            dd_variance,
            datasource or "A",
            comparison_dd_variance,
            comparison_datasource or "B",
            return_fig=True,
        )

        if show_descriptive_stats:
            mean1, std1 = dd_variance.mean(), dd_variance.std()
            mean2, std2 = comparison_dd_variance.mean(), comparison_dd_variance.std()

            u_stat, p_value = mannwhitneyu(
                dd_variance, comparison_dd_variance, alternative="two-sided"
            )

            stats_text = (
                f"{format_applied_filters(datasource, survey.applied_filters, max_words=95)} Mean: {mean1:.2f}, Std. Dev.: {std1:.2f}<br>"
                f"{format_applied_filters(comparison_datasource, comparison_survey.applied_filters, max_words=95)} Mean: {mean2:.2f}, Std. Dev.: {std2:.2f}<br>"
                f"Mann-Whitney U: {u_stat}, p-value: {p_value:.4f}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.5,
                showarrow=False,
                text=stats_text,
                align="left",
                font=dict(size=12),
                borderwidth=1,
                borderpad=4,
            )

            fig.update_layout(margin=dict(b=120))

        st.plotly_chart(fig, use_container_width=True)
    else:
        plot_single(
            st,
            dd_variance,
            "Distribution of Variance in Delay Discounting",
            "histogram",
        )


def display_delay_discounting_k_values(
    st,
    survey: Survey,
    comparison_survey: Survey = None,
    datasource=None,
    comparison_datasource=None,
    show_descriptive_stats=False,
) -> None:
    """Displays a histogram plot of the k values in delay discounting responses within the survey data.

    Args:
        st: The Streamlit object used to render the plot.
        survey: The Survey instance containing the survey data and metadata.
        comparison_survey: An optional second Survey instance for comparison.
        datasource: The name of the first data source.
        comparison_datasource: The name of the second data source.
        show_descriptive_stats: Whether to display descriptive statistics for the data.
    """

    k_values = survey.get_k_values()
    if comparison_survey:
        comparison_k_values = comparison_survey.get_k_values()
        fig = plot_side_by_side(
            st,
            "Distribution of k Values in Delay Discounting Responses",
            k_values,
            datasource or "A",
            comparison_k_values,
            comparison_datasource or "B",
            return_fig=True,
        )

        if show_descriptive_stats:
            mean1, std1 = k_values.mean(), k_values.std()
            mean2, std2 = comparison_k_values.mean(), comparison_k_values.std()

            u_stat, p_value = mannwhitneyu(
                k_values, comparison_k_values, alternative="two-sided"
            )

            stats_text = (
                f"{format_applied_filters(datasource, survey.applied_filters, max_words=95)} Mean: {mean1:.2f}, Std. Dev.: {std1:.2f}<br>"
                f"{format_applied_filters(comparison_datasource, comparison_survey.applied_filters, max_words=95)} Mean: {mean2:.2f}, Std. Dev.: {std2:.2f}<br>"
                f"Mann-Whitney U: {u_stat}, p-value: {p_value:.4f}"
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.5,
                showarrow=False,
                text=stats_text,
                align="left",
                font=dict(size=12),
                borderwidth=1,
                borderpad=4,
            )

            fig.update_layout(margin=dict(b=120))

        st.plotly_chart(fig, use_container_width=True)
    else:
        plot_single(
            st,
            k_values,
            "Distribution of k Values in Delay Discounting Responses",
            "histogram",
        )


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
        title = split_text(title)

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
