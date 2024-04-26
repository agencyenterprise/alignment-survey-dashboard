import argparse
import pandas as pd
from email.message import EmailMessage
from survey import Survey
import plots
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from constants import ALIGNMENT_QUESTION_PAIRS, EA_QUESTION_PAIRS, SCORING_MAPPING
from typing import List, BinaryIO


def _get_survey(dataset: str) -> Survey:
    """Retrieve a survey object based on the dataset name.

    Args:
        dataset (str): The name of the dataset ('ea' or 'alignment').

    Returns:
        Survey: A survey object loaded with the corresponding dataset.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if dataset == "ea":
        return Survey.from_file("ea_data.csv")
    elif dataset == "alignment":
        return Survey.from_file("alignment_data.csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _get_survey_with_emails(survey: Survey, original_csv_path: str) -> Survey:
    """Merge emails into the survey data from a CSV file.

    Args:
        survey (Survey): The survey object to which emails will be merged.
        original_csv_path (str): The path to the original CSV file containing emails.

    Returns:
        Survey: The updated survey object with emails merged.
    """
    with open(original_csv_path, "rb") as file:
        df = pd.read_csv(file)
        email_col = df.columns[
            df.columns.str.contains("please put your email below", case=False)
        ]
        df = df.iloc[2:]

        emails = df[email_col]
        emails.columns = ["Email"]
        survey.merge_emails(emails)
    return survey


def _get_plot_images(survey: Survey, email: str) -> List[ImageReader]:
    """Generate plot images from the survey data, highlighting the value corresponding to a specific email.

    Args:
        survey (Survey): The survey object containing the data to be plotted.
        email (str): The email address used to identify the specific row to highlight.

    Returns:
        List[ImageReader]: A list of ImageReader objects for the generated plot images.
    """
    images = []
    data = survey.get_numeric_data()
    highlight_value = None

    highlight_value = survey.data.loc[survey.data["Email"] == email]
    mini_survey = Survey(data=highlight_value, metadata=survey.metadata)
    mini_data = mini_survey.get_numeric_data()

    for column in data.columns:
        if column.startswith("Overall Distribution for"):
            plot_kwargs = {
                "nbins": 10,
                "highlight_value": mini_data[column].iloc[0],
                "highlight_color": "red",
            }
            fig = plots.plot_single(
                None, data[column], column, plot_kwargs=plot_kwargs, return_fig=True
            )
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            images.append(ImageReader(buf))

    for column in data.columns:
        if column in (ALIGNMENT_QUESTION_PAIRS | EA_QUESTION_PAIRS).keys():
            formatted_data = plots.format_survey_data_for_plotting(survey, column)
            plot_kwargs = {}
            if highlight_value is not None and column in highlight_value:
                plot_kwargs["highlight_value"] = plots.format_survey_data_for_plotting(mini_survey, column).iloc[0]
                plot_kwargs["highlight_color"] = "red"

            fig = plots.plot_single(
                None,
                formatted_data,
                column,
                plot_kwargs=plot_kwargs,
                zoom_to_fit_categories=True,
                return_fig=True,
            )
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            images.append(ImageReader(buf))

    return images


def _create_pdf(images: List[ImageReader]) -> BinaryIO:
    """Create a PDF document from a list of images, with a title "Survey Results" and an introductory paragraph on the first page, and page numbers on each page.

    Args:
        images (List[ImageReader]): The images to include in the PDF.

    Returns:
        BinaryIO: A binary stream containing the generated PDF document.
    """
    margin_left = 50
    margin_top = 25
    pdf_bytes = io.BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)
    max_width, max_height = c._pagesize

    # PDF Title (only on the first page)
    title = "Survey Results"
    title_font_size = 18
    c.setFont("Helvetica-Bold", title_font_size)
    c.drawString(margin_left, max_height - margin_top - title_font_size, title)

    intro_text = (
        "This document contains the results of the survey. "
        "The following charts and analyses are based on the collected responses. "
        "Please review the data for insights into the survey outcomes."
    )
    intro_font_size = 12
    c.setFont("Times-Roman", intro_font_size)

    # Splitting the introductory text into multiple lines
    intro_lines = intro_text.split(". ")
    current_intro_height = (
        max_height - margin_top - title_font_size - 40
    )  # Starting height for the intro text

    for line in intro_lines:
        c.drawString(margin_left, current_intro_height, line + ".")
        current_intro_height -= intro_font_size + 5  # Move to the next line

    current_height = max_height - current_intro_height + margin_top + 10
    max_width -= 2 * margin_left
    max_height -= 2 * margin_top

    page_number = 1

    for image in images:
        width, height = image.getSize()

        if width > max_width:
            scale_factor = max_width / width
            width *= scale_factor
            height *= scale_factor

        if current_height + height > max_height:
            c.drawString(max_width / 2 + margin_left, 10, str(page_number))
            page_number += 1
            c.showPage()
            current_height = 0
            max_height = c._pagesize[1] - 2 * margin_top

        c.drawImage(
            image,
            x=margin_left,
            y=max_height - current_height - height,
            width=width,
            height=height,
        )
        current_height += height + 10

    # Draw page number on the last page
    c.drawString(max_width / 2, 10, str(page_number))

    c.showPage()
    c.save()
    pdf_bytes.seek(0)
    return pdf_bytes


def _create_email(pdf_bytes: BinaryIO, email_address: str) -> None:
    """Create and save an email with a PDF attachment.

    Args:
        pdf_bytes (BinaryIO): The binary stream of the PDF to attach.
        email_address (str): The email address to which the email will be saved.

    Returns:
        None
    """
    msg = EmailMessage()
    msg["Subject"] = "Your Subject Here"
    msg["From"] = "cameron@ae.studio"
    msg["To"] = email_address
    msg.set_content("This is the body of the email.")

    msg.add_attachment(
        pdf_bytes.read(),
        maintype="application",
        subtype="pdf",
        filename="report.pdf",
    )

    os.makedirs("emails", exist_ok=True)
    with open(f"emails/{email_address}.eml", "w") as eml_file:
        eml_file.write(msg.as_string())


def main(csv_file_path: str, dataset: str) -> None:
    """The main function to process survey data and create emails with PDF attachments.

    Args:
        csv_file_path (str): The path to the CSV file containing survey data and emails.
        dataset (str): The name of the dataset to process ('ea' or 'alignment').

    Returns:
        None
    """
    survey = _get_survey(dataset)
    survey = _get_survey_with_emails(survey, csv_file_path)

    for email in survey.data["Email"].dropna():
        images = _get_plot_images(survey, email)
        pdf_bytes = _create_pdf(images)
        _create_email(pdf_bytes, email)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an EML file with a CSV attachment."
    )
    parser.add_argument(
        "--original", help="Path to the original CSV file to attach.", required=True
    )
    parser.add_argument(
        "--dataset",
        help="Dataset to use: 'ea' or 'alignment'.",
        required=True,
        choices=["ea", "alignment"],
    )

    args = parser.parse_args()

    main(args.original, args.dataset)
