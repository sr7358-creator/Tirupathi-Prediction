# TTD Darshan Forecast (v2)

Single-page Streamlit app. Forecasts **tomorrow's crowd, tonsures and Sarvadarshanam waiting time**
from an Excel file. There is no data-entry page: you update the Excel file and refresh the app.

## Run
```
pip install -r requirements.txt
streamlit run app.py
python evaluate.py        # regenerates the ablation tables in reports/
```
Delete the old `pages/` folder (the Data Entry page is no longer used).

## Updating the data (`data/ttd_data.xlsx`)
* **Data sheet** - one row per date, real Excel dates, numbers as numbers. Leave a cell empty if unknown
  (never 0). Laddu / Annaprasadam / Medical Treatment are stored but not yet used by the models.
* **Events sheet** - festivals (`festival`) and holidays (`holiday`) with Start/End dates. Add upcoming
  ones *before* they happen. Rows marked "verify" should be checked against the official TTD calendar.
* Save and close Excel before refreshing the app. The app retrains only when the file changes.

## Method
* Full daily calendar, duplicates removed, **targets never imputed**, no `dropna()` over unused columns.
* Wait time is grouped into 5 announced tiers (<=8, 9-12, 13-18, 19-24, >24 hrs).
* **Covariate Markov chain**: P(tomorrow's tier | today's tier, weekday, festival/holiday flags,
  compartments, crowd, tonsures, season) via multinomial logistic regression.
* Headline wait time = average of an XGBoost regression and the Markov chain's median.
* Crowd / tonsures: XGBoost predicting the change from today.
* All accuracy numbers are **walk-forward** (train on the past, test on the next 30 days, repeat).

## Hosting
The Excel file lives in the repo, so it is read-only when hosted: to update, replace the file on GitHub.

## Structure
```
app.py                    Forecast page
evaluate.py               Ablation study for the report
utils/model_pipeline.py   Loading, features, Markov + XGBoost models, walk-forward evaluation
data/ttd_data.xlsx        Data, Events, Instructions sheets
reports/*.csv             Ablation results
```
