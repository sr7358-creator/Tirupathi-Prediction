import os
import sys
import pandas as pd
import streamlit as st

try:                                    # friendly message if launched with `python app.py`
    from streamlit.runtime import exists as _st_running
    if not _st_running():
        print("\nThis is a Streamlit app. Open a terminal IN THE PROJECT FOLDER and run:\n"
              "    streamlit run app.py\n")
        sys.exit(0)
except ImportError:
    pass

from utils.model_pipeline import train_and_forecast

st.set_page_config(page_title="TTD Darshan Forecast", page_icon="🛕", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "ttd_data.xlsx")
ABLATION_PATH = os.path.join(BASE, "reports", "ablation_wait.csv")


@st.cache_data(show_spinner=False)
def run(path, mtime):
    """mtime is part of the cache key, so the models retrain only when the Excel file changes."""
    return train_and_forecast(path)


st.title("🛕 TTD Darshan Forecast")
st.caption("Next-day forecast of crowd size, tonsures and Sarvadarshanam waiting time. "
           "Data is read from `data/ttd_data.xlsx`; update the Excel file to refresh the forecast.")

if not os.path.exists(DATA_PATH):
    st.error("Data file not found at `data/ttd_data.xlsx`.")
    st.stop()

with st.spinner("Training models on the latest data..."):
    try:
        r = run(DATA_PATH, os.path.getmtime(DATA_PATH))
    except PermissionError:
        st.error("The Excel file is open in another program. Save and close it, then refresh this page.")
        st.stop()
    except Exception as e:
        st.error(f"Couldn't train the models: {e}")
        st.stop()

# ---- data-freshness warnings
if r["days_stale"] > 1:
    st.warning(f"Latest entry in the Excel file is **{r['last_date']:%d %b %Y}** ({r['days_stale']} days ago), "
               f"so this forecast is for {r['forecast_date']:%d %b %Y}, not for tomorrow. "
               "Add the missing days to the Data sheet for a current forecast.")
if pd.notna(r["events_end"]) and r["forecast_date"] > r["events_end"]:
    st.info(f"The Events sheet ends on {r['events_end']:%d %b %Y}. Add upcoming festivals and holidays "
            "there so the model knows about them.")

st.subheader(f"Forecast for {r['forecast_date']:%d %b %Y}")
st.caption(f"Based on data through {r['last_date']:%d %b %Y} ({r['n_days']:,} days of history). "
           f"Today's waiting-time tier: **{r['wait_today_tier']}**.")

wsum, ce, te = r["wait_summary"], r["crowd_eval"], r["tonsures_eval"]
c1, c2, c3 = st.columns(3)
c1.metric("Predicted Crowd Size", f"{r['crowd_forecast']:,.0f} people",
          help=f"Typical error: ±{ce.loc['model', 'MAE']:,.0f} people")
c2.metric("Predicted Tonsures", f"{r['tonsures_forecast']:,.0f}",
          help=f"Typical error: ±{te.loc['model', 'MAE']:,.0f}")
c3.metric("Predicted Waiting Time", f"{r['wait_forecast']:.1f} hrs",
          help=f"Typical error: ±{wsum.loc[r['wait_method'], 'MAE_hrs']:.1f} hrs")

st.markdown("**Waiting-time tier probabilities** (covariate Markov chain)")
probs = pd.Series(r["wait_probs"], name="Probability")
st.bar_chart(probs * 100, y_label="Probability (%)", height=220)
top = probs.idxmax()
st.caption(f"Most likely tier: **{top}** ({probs.max() * 100:.0f}%). "
           "Wait times are announced in tiers, so the probabilities show how confident the model is.")

st.divider()
st.subheader(f"Model accuracy (walk-forward test on the last {r['wait_eval_days']} days)")
st.caption("Each day is predicted by a model trained only on earlier days, so these numbers are honest "
           "out-of-sample results, compared with a naive 'tomorrow = today' guess.")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Crowd accuracy", f"{ce.loc['model', 'accuracy_%']:.1f}%",
          delta=f"{ce.loc['model', 'accuracy_%'] - ce.loc['persistence', 'accuracy_%']:+.1f} pts vs naive")
a2.metric("Tonsures accuracy", f"{te.loc['model', 'accuracy_%']:.1f}%",
          delta=f"{te.loc['model', 'accuracy_%'] - te.loc['persistence', 'accuracy_%']:+.1f} pts vs naive")
m, p = wsum.loc[r["wait_method"]], wsum.loc["persistence"]
a3.metric("Wait-time error (MAE)", f"{m['MAE_hrs']:.2f} hrs", delta=f"{m['MAE_hrs'] - p['MAE_hrs']:+.2f} vs naive",
          delta_color="inverse")
a4.metric("Wait time within ±2 hrs", f"{m['within_2hr_%']:.0f}% of days",
          delta=f"{m['within_2hr_%'] - p['within_2hr_%']:+.0f} pts vs naive",
          help=f"Within ±3 hrs: {m['within_3hr_%']:.0f}% of days. Trained on {r['n_wait_train']:,} days.")

if os.path.exists(ABLATION_PATH):
    with st.expander("How the waiting-time method was chosen (ablation study)"):
        st.caption("All methods scored on the same 2024-07 to 2026-08 walk-forward days. "
                   "Chosen: **hybrid_reg+logit** = average of an XGBoost regression and the covariate "
                   "Markov chain's median. Run `python evaluate.py` to regenerate.")
        st.dataframe(pd.read_csv(ABLATION_PATH), hide_index=True)

st.divider()
st.subheader("Recent history")
h = r["history"].tail(60)
t1, t2 = st.tabs(["Crowd & Tonsures", "Waiting Time"])
with t1:
    st.line_chart(h[["Darshned People", "Tonsures"]])
with t2:
    st.line_chart(h[["Approx. Darsan Time for Sarvadarshanam"]].rename(
        columns={"Approx. Darsan Time for Sarvadarshanam": "Wait Time (hrs)"}))
