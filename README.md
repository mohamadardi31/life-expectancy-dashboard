# Life Expectancy Analytics Dashboard

Professional Streamlit healthcare analytics dashboard for exploring and predicting population health outcomes using a WHO-style Life Expectancy dataset.

## Project Description

This project analyzes how life expectancy differs across countries, years, and development status, and how it relates to mortality, disease burden, immunization, health expenditure, GDP, income composition, and schooling.

Main research question:

> Which health, healthcare system, and socioeconomic indicators are most strongly associated with life expectancy across countries from 2000 to 2015?

## Dataset Requirements

The app supports CSV and Excel files with columns similar to:

- Country
- Year
- Status
- Life expectancy
- Adult Mortality
- infant deaths
- Alcohol
- percentage expenditure
- Hepatitis B
- Measles
- BMI
- under-five deaths
- Polio
- Total expenditure
- Diphtheria
- HIV/AIDS
- GDP
- Population
- thinness 1-19 years
- thinness 5-9 years
- Income composition of resources
- Schooling

The code automatically cleans column names, handles extra spaces and inconsistent capitalization, converts numeric fields, imputes missing predictors, and removes records missing life expectancy.

## How To Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard password:

```text
healthanalytics2026
```

The project includes a local copy of the dataset in `data/Life Expectancy Data.csv` so the dashboard can run immediately. Users can also upload a different CSV or Excel file from the sidebar.

## Dashboard Pages

1. Executive Overview
2. Trends Over Time
3. Mortality & Disease Burden
4. Healthcare System Indicators
5. Socioeconomic Drivers
6. Global Map
7. Predictive Model
8. Data Dictionary & Methodology

## Predictive Model

The model predicts Life Expectancy using available numeric indicators and one-hot encoded Status when present. It uses a RandomForestRegressor as the main model and LinearRegression as a comparison. The dashboard reports:

- R-squared
- RMSE
- MAE
- Actual vs predicted chart
- Residual chart
- Top 10 feature importances
- Interactive prediction form

## Deployment On Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Upload `app.py`, `requirements.txt`, `README.md`, `CONSULTANT_REPORT.md`, and the `data` folder.
3. Go to Streamlit Community Cloud.
4. Select the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy the app.
7. Open the deployed link and enter the password.

## Suggested Folder Structure

```text
streamlit HA project/
  app.py
  requirements.txt
  README.md
  CONSULTANT_REPORT.md
  data/
    Life Expectancy Data.csv
```
