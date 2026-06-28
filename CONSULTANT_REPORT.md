# Consultant Manual / Report Draft

## 1. Executive Summary

This project presents a healthcare analytics dashboard for understanding life expectancy patterns across countries from 2000 to 2015. The dashboard helps public health leaders examine differences by country, year, and development status, while also identifying health system and socioeconomic indicators associated with better or worse population health outcomes.

The main research question is: Which health, healthcare system, and socioeconomic indicators are most strongly associated with life expectancy across countries from 2000 to 2015?

Key analytical themes include adult mortality, child mortality, HIV/AIDS burden, immunization coverage, health expenditure, GDP, income composition, and schooling.

## 2. Data Sources and Variables

The project uses a WHO-style Life Expectancy dataset with country-year records. The dataset includes:

- Outcome variable: Life Expectancy
- Mortality indicators: Adult Mortality, Infant Deaths, Under-Five Deaths
- Disease burden indicators: HIV/AIDS, Measles, BMI, Thinness Age 1-19, Thinness Age 5-9
- Healthcare system indicators: Hepatitis B, Polio, Diphtheria, Percentage Health Expenditure, Total Expenditure
- Socioeconomic indicators: GDP, Population, Income Composition, Schooling
- Context variables: Country, Year, Status

The data is cleaned automatically by standardizing column names, converting numeric fields, removing records without life expectancy, imputing missing numeric predictors with median values, and filling missing categorical values with Unknown.

## 3. Dashboard Components

The dashboard is password protected and organized into eight pages:

1. Executive Overview: KPI cards, top and bottom country comparisons, status distribution, and adult mortality scatterplot.
2. Trends Over Time: Life expectancy, adult mortality, schooling, and status-group trends by year.
3. Mortality & Disease Burden: Mortality and disease relationships with life expectancy, including correlation heatmaps.
4. Healthcare System Indicators: Immunization and expenditure relationships with life expectancy.
5. Socioeconomic Drivers: GDP, schooling, income composition, population, and development status analysis.
6. Global Map: Choropleth map and ranking tables for selected health or socioeconomic metrics.
7. Predictive Model: Random Forest model, performance metrics, residuals, feature importance, and interactive prediction form.
8. Data Dictionary & Methodology: Definitions, cleaning steps, filters, model explanation, limitations, and suggested use.

## 4. Key Findings

Expected findings from this dataset include:

- Adult mortality generally has a strong negative association with life expectancy.
- Countries with higher schooling and income composition usually report higher life expectancy.
- HIV/AIDS burden is often associated with lower life expectancy in affected country-year records.
- Immunization coverage, especially Polio and Diphtheria coverage, is generally positively associated with life expectancy.
- Developed countries tend to have higher life expectancy distributions than developing countries, although individual country patterns vary.

These findings should be presented as associations rather than causal proof.

## 5. Predictive Model

The dashboard uses a Random Forest regression model to predict Life Expectancy. Available numeric predictors are used dynamically, and Status is one-hot encoded when present. The model workflow includes:

- Train-test split
- Median imputation for numeric features
- One-hot encoding for Status
- Random Forest training
- Linear Regression comparison
- Evaluation with R-squared, RMSE, and MAE
- Actual vs predicted visualization
- Residual visualization
- Feature importance ranking
- Interactive user prediction form

The model is useful for identifying important predictors and estimating likely life expectancy under selected country-level conditions. It should not be used as a causal model or patient-level clinical tool.

## 6. Limitations

1. The data is country-level, not patient-level.
2. Relationships are correlations, not proof of causation.
3. Missing values and reporting differences between countries may affect results.
4. Some countries may have incomplete data across years.
5. National averages can hide inequality within regions, income groups, and health systems.

## 7. Recommendations

Public health decision-makers can use the dashboard to:

- Identify countries and years with lower life expectancy outcomes.
- Investigate mortality and disease burden patterns that may require targeted intervention.
- Review immunization coverage gaps and compare them with health outcomes.
- Explore socioeconomic drivers such as schooling and income composition.
- Use model feature importance as a starting point for deeper policy or program review.

Recommended next steps include validating findings with local health system data, combining country-level insights with regional or patient-level sources when available, and using the dashboard as an exploratory decision-support tool for public health planning.
