# Alignment Survey Dashboard

This repository contains the code associated with our project, [Key takeaways from our EA and alignment research surveys](https://www.lesswrong.com/posts/XTdByFM6cmgB3taEN/key-takeaways-from-our-ea-and-alignment-research-surveys). Here, you will find the anonymized data from both surveys, as well as the code that is running the data analysis and visualization tool described in the write-up.

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
