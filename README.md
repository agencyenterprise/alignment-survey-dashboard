# Alignment Survey Dashboard

The app is automatically deployed to https://alignment-survey-dashboard.streamlit.app/ upon push to main and also available at https://thealignmentsurvey.com/ which presents the content from  https://alignment-survey-dashboard.streamlit.app/ in an iframe.

## Running the app locally

Install the requirements:

```
pip install -r requirements.txt
```

And start it via:

```
streamlit run app.py 
```

## Generating emails for the survey participants

Install the requirements:

```
pip install -r requirements.txt
```

_Make sure you have a local copy of the original csv files: `alignment_data_original.csv` and `ea_data_original.csv` which are not included in this repo because of personal identifiable information. The files included in this repo (`alignment_data.csv` and `ea_data.csv`) have extra annotations compared to original files and the email addresses were removed._

The original csv files are needed to generate emails. 

Run the script for the EA dataset:

```
python generate_emails.py --original ea_data_original.csv --dataset ea
```

Run the script for the Alignment dataset:

```
python generate_emails.py --original alignment_data_original.csv --dataset alignment
```

The emails will be generated in the `emails` folder.