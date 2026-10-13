"""
TTD forecast pipeline v2
- Reads data + festival calendar from Excel (no dates hard-coded here)
- Works on a full daily calendar; targets are never imputed
- Wait time: 5 "tier" states + covariate-dependent Markov models
- Honest walk-forward evaluation shared by evaluate.py and the app
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLS = {'Darshned People': 'crowd', 'Tonsures': 'tons', 'Hundi Kanakulu': 'hundi',
        'Waiting Comaprtments': 'comp', 'Approx. Darsan Time for Sarvadarshanam': 'wait'}
ALIASES = {'Waiting Compartments': 'Waiting Comaprtments'}   # accept the correct spelling too

TIER_EDGES = [8, 12, 18, 24]                                   # <=8 | 9-12 | 13-18 | 19-24 | >24
TIER_NAMES = ['≤8 hrs', '9–12 hrs', '13–18 hrs', '19–24 hrs', '>24 hrs']
N_TIERS = 5
DEFAULT_TIER_HOURS = [6, 11, 16, 22, 30]
WAIT_START = pd.Timestamp('2023-01-01')      # wait readings are reliable from 2023
LEVEL_START = pd.Timestamp('2022-01-01')     # crowd / tonsure training start
WAIT_METHOD = 'hybrid_reg+logit'             # headline wait time (chosen by the ablation, reports/ablation_wait.csv)
WAIT_PROB_MODEL = 'markov_logit'             # covariate Markov chain -> tier probabilities shown in the app


def tier(h):
    return np.digitize(np.asarray(h, dtype=float), TIER_EDGES, right=True)


# --------------------------------------------------------------------------- data
def load_data(path):
    """Returns (daily dataframe with short column names, events dataframe)."""
    raw = pd.read_excel(path, sheet_name='Data')
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.rename(columns=ALIASES)
    missing = [c for c in ['Date', *COLS] if c not in raw.columns]
    if missing:
        raise ValueError(f"Data sheet is missing column(s): {missing}")
    df = raw[['Date', *COLS]].rename(columns=COLS)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    for c in COLS.values():
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
    df = df.dropna(subset=['Date']).drop_duplicates('Date', keep='last').set_index('Date').sort_index()
    if len(df) < 200:
        raise ValueError("Need at least ~200 days of history to train.")
    df = df.asfreq('D')                                       # full daily calendar (gaps = NaN)

    try:
        ev = pd.read_excel(path, sheet_name='Events')
        ev.columns = [str(c).strip().lower() for c in ev.columns]
        ev['start'] = pd.to_datetime(ev['start'], dayfirst=True, errors='coerce')
        ev['end'] = pd.to_datetime(ev['end'], dayfirst=True, errors='coerce')
        ev['type'] = ev['type'].astype(str).str.strip().str.lower()
        ev = ev.dropna(subset=['start', 'end'])
    except ValueError:                                        # no Events sheet
        ev = pd.DataFrame(columns=['start', 'end', 'event', 'type'])
    return df, ev


def _flags(index, ev, kind):
    f = pd.Series(0.0, index=index)
    for r in ev[ev['type'] == kind].itertuples():
        f[(index >= r.start) & (index <= r.end)] = 1.0
    return f


def build_features(df, ev):
    """
    Row t = everything known at the end of day t. Targets = day t+1.
    The last real day keeps its features (target NaN) -> that row is the live forecast input.
    """
    ext = pd.date_range(df.index[0], df.index[-1] + pd.Timedelta(days=1))
    d = df.reindex(ext)
    fidx = pd.date_range(ext[0], ext[-1] + pd.Timedelta(days=3))
    fest, hol = _flags(fidx, ev, 'festival'), _flags(fidx, ev, 'holiday')

    X = pd.DataFrame(index=ext)
    for c in ['crowd', 'tons', 'hundi', 'comp']:
        X[c] = d[c]
    X['wait0'] = d['wait'].ffill(limit=3)              # today's wait (carried forward max 3 days)
    X['wait_l1'] = d['wait'].shift(1)
    X['wait_l6'] = d['wait'].shift(6)                  # same weekday as tomorrow, last week
    X['wait_r3'] = d['wait'].rolling(3, min_periods=1).mean()
    X['wait_r7'] = d['wait'].rolling(7, min_periods=2).mean()
    X['crowd_l6'] = d['crowd'].shift(6)
    X['crowd_r7'] = d['crowd'].rolling(7, min_periods=3).mean()
    X['tons_l6'] = d['tons'].shift(6)
    X['comp_full'] = (d['comp'] >= 31).astype(float).where(d['comp'].notna())
    X['flush'] = (d['comp'] >= 22).astype(float).where(d['comp'].notna())
    X['crowd_per_comp'] = d['crowd'] / (d['comp'] + 1)
    X['fest_t'] = fest.reindex(ext)
    X['hol_t'] = hol.reindex(ext)
    X['fest_t1'] = fest.shift(-1).reindex(ext)
    X['fest_t2'] = fest.shift(-2).reindex(ext)
    X['hol_t1'] = hol.shift(-1).reindex(ext)
    X['dow_t1'] = (ext.dayofweek + 1) % 7
    X['weekend_t1'] = (X['dow_t1'] >= 5).astype(float)
    m1 = (ext + pd.Timedelta(days=1)).month
    X['month_sin'] = np.sin(2 * np.pi * m1 / 12)
    X['month_cos'] = np.cos(2 * np.pi * m1 / 12)

    Y = pd.DataFrame({'crowd': d['crowd'].shift(-1), 'tons': d['tons'].shift(-1),
                      'wait': d['wait'].shift(-1)}, index=ext)
    last = df.index[-1]
    return X.loc[:last], Y.loc[:last], last


LOGIT_NUM = ['comp', 'crowd', 'tons', 'hundi', 'wait0', 'wait_r3', 'wait_r7', 'crowd_l6',
             'comp_full', 'fest_t', 'hol_t', 'fest_t1', 'fest_t2', 'hol_t1', 'weekend_t1',
             'month_sin', 'month_cos']


def _logit_matrix(X):
    M = X[LOGIT_NUM].copy()
    s0, dw = tier(X['wait0'].fillna(12).values), X['dow_t1'].values
    for k in range(N_TIERS):
        M[f'state{k}'] = (s0 == k).astype(float)
    for k in range(7):
        M[f'dow{k}'] = (dw == k).astype(float)
    return M


def _pad(p, classes):
    full = np.zeros((len(p), N_TIERS))
    full[:, np.asarray(classes, dtype=int)] = p
    return full


# --------------------------------------------------------------------------- wait models
class WaitModels:
    """Persistence, XGBoost regression, plain Markov, and covariate-dependent Markov chains."""

    def fit(self, X, y):
        self.cols = list(X.columns)
        st = tier(y.values)
        self.hours = np.array([np.median(y.values[st == k]) if (st == k).any() else DEFAULT_TIER_HOURS[k]
                               for k in range(N_TIERS)])
        s0 = tier(X['wait0'].values)
        C = np.full((N_TIERS, N_TIERS), 0.5)                    # plain Markov (Laplace smoothed)
        for a, b in zip(s0, st):
            C[a, b] += 1
        self.P = C / C.sum(1, keepdims=True)

        self.logit = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                   LogisticRegression(C=0.3, max_iter=3000)).fit(_logit_matrix(X), st)
        self.cls = np.unique(st)
        self.xclf = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.8, min_child_weight=3, random_state=42,
                                      objective='multi:softprob')
        self.xclf.fit(X, np.searchsorted(self.cls, st))
        self.reg = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=5, random_state=42,
                                    objective='reg:absoluteerror').fit(X, y)
        self.regd = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=5, random_state=42,
                                     objective='reg:absoluteerror').fit(X, y - X['wait0'])   # predicts the CHANGE
        return self

    def predict(self, X):
        s0 = tier(X['wait0'].values)
        probs = {'markov_plain': self.P[s0],
                 'markov_logit': _pad(self.logit.predict_proba(_logit_matrix(X)), self.logit.classes_),
                 'markov_xgb': _pad(self.xclf.predict_proba(X[self.cols]), self.cls)}
        probs['markov_blend'] = (probs['markov_logit'] + probs['markov_xgb']) / 2
        hours = {'persistence': X['wait0'].values,
                 'xgb_reg': np.clip(self.reg.predict(X[self.cols]), 0, 48)}
        hours['xgb_delta'] = np.clip(X['wait0'].values + self.regd.predict(X[self.cols]), 0, 48)
        for k, p in probs.items():
            hours[k + '_median'] = self.hours[(np.cumsum(p, 1) >= 0.5).argmax(1)]
            hours[k + '_mode'] = self.hours[p.argmax(1)]
            hours[k + '_expected'] = p @ self.hours
        hours['hybrid_reg+logit'] = (hours['xgb_reg'] + hours['markov_logit_median']) / 2
        hours['hybrid_reg+logit_exp'] = (hours['xgb_reg'] + hours['markov_logit_expected']) / 2
        hours['hybrid_reg+delta'] = (hours['xgb_reg'] + hours['xgb_delta']) / 2
        return hours, probs, s0


def _wait_frame(X, Y):
    m = Y['wait'].notna() & X['wait0'].notna() & (X.index >= WAIT_START)
    return X[m], Y.loc[m, 'wait']


def evaluate_wait(X, Y, start='2024-07-01', block=30):
    """Expanding-window walk-forward. Returns per-day predictions of every method vs actual."""
    D, y = _wait_frame(X, Y)
    rows = []
    for a in pd.date_range(start, D.index.max(), freq=f'{block}D'):
        tr, te = D.index < a, (D.index >= a) & (D.index < a + pd.Timedelta(days=block))
        if te.sum() == 0 or tr.sum() < 200:
            continue
        hours, probs, s0 = WaitModels().fit(D[tr], y[tr]).predict(D[te])
        out = pd.DataFrame(hours, index=D.index[te])
        out['actual'] = y[te]
        out['state_actual'] = tier(y[te].values)
        for k, p in probs.items():
            out[k + '_state'] = p.argmax(1)
        rows.append(out)
    return pd.concat(rows)


def summarize_wait(R):
    rows = []
    for name in [c for c in R.columns if c not in ('actual', 'state_actual') and not c.endswith('_state')]:
        err = (R['actual'] - R[name]).abs()
        base = name.rsplit('_', 1)[0]
        sa = (R[base + '_state'] == R['state_actual']).mean() * 100 if base + '_state' in R else np.nan
        rows.append({'method': name, 'MAE_hrs': err.mean(), 'within_2hr_%': (err <= 2).mean() * 100,
                     'within_3hr_%': (err <= 3).mean() * 100, 'tier_accuracy_%': sa, 'n_days': len(err)})
    return pd.DataFrame(rows).sort_values('MAE_hrs').reset_index(drop=True)


# --------------------------------------------------------------------------- crowd / tonsures
def _level_frame(X, Y, col):
    m = Y[col].notna() & X[col].notna() & (X.index >= LEVEL_START)
    return X[m], Y.loc[m, col]


def _fit_level(X, y, col, delta):
    reg = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.04, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=5, random_state=42,
                           objective='reg:absoluteerror')
    reg.fit(X, (y - X[col]) if delta else y)
    return reg


def _pred_level(reg, X, col, delta):
    p = reg.predict(X) + (X[col].values if delta else 0)
    return np.maximum(p, 0)


def evaluate_level(X, Y, col, start='2025-01-01', block=30, delta=False):
    D, y = _level_frame(X, Y, col)
    rows = []
    for a in pd.date_range(start, D.index.max(), freq=f'{block}D'):
        tr, te = D.index < a, (D.index >= a) & (D.index < a + pd.Timedelta(days=block))
        if te.sum() == 0:
            continue
        reg = _fit_level(D[tr], y[tr], col, delta)
        rows.append(pd.DataFrame({'model': _pred_level(reg, D[te], col, delta),
                                  'persistence': D.loc[te, col].values,
                                  'last_week': D.loc[te, col + '_l6'].values if col + '_l6' in D else np.nan,
                                  'actual': y[te].values}, index=D.index[te]))
    return pd.concat(rows)


def summarize_level(R):
    out = {}
    for m in ['model', 'persistence', 'last_week']:
        e = (R['actual'] - R[m]).abs().dropna()
        out[m] = {'MAE': e.mean(), 'accuracy_%': (1 - (e / R['actual'][e.index]).mean()) * 100}
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------- live forecast
def train_and_forecast(path, eval_days=365):
    df, ev = load_data(path)
    X, Y, last = build_features(df, ev)
    fdate = last + pd.Timedelta(days=1)
    x_live = X.loc[[last]]
    if pd.isna(x_live['wait0'].iloc[0]):
        raise ValueError("No waiting-time reading in the last 4 days - add recent wait times to the Data sheet.")

    # wait time: final model on all data, live probabilities for tomorrow
    D, y = _wait_frame(X, Y)
    wm = WaitModels().fit(D, y)
    hours, probs, s0 = wm.predict(x_live)
    method = WAIT_METHOD
    wait_fc = float(hours[method][0])
    p_live = probs[WAIT_PROB_MODEL][0]

    # honest accuracy on the most recent `eval_days`
    start = (last - pd.Timedelta(days=eval_days)).strftime('%Y-%m-%d')
    Rw = evaluate_wait(X, Y, start=start)
    summ = summarize_wait(Rw).set_index('method')

    out = {}
    for col, key in [('crowd', 'crowd'), ('tons', 'tonsures')]:
        Dl, yl = _level_frame(X, Y, col)
        reg = _fit_level(Dl, yl, col, delta=True)
        out[key + '_forecast'] = float(_pred_level(reg, x_live, col, True)[0])
        out[key + '_eval'] = summarize_level(evaluate_level(X, Y, col, start=start, delta=True))

    hist = df.rename(columns={v: k for k, v in COLS.items()})
    return {
        'last_date': last, 'forecast_date': fdate, 'days_stale': (pd.Timestamp.today().normalize() - last).days,
        'n_days': int(df['crowd'].notna().sum()), 'n_wait_train': len(D),
        'crowd_forecast': out['crowd_forecast'], 'tonsures_forecast': out['tonsures_forecast'],
        'crowd_eval': out['crowd_eval'], 'tonsures_eval': out['tonsures_eval'],
        'wait_forecast': wait_fc, 'wait_probs': dict(zip(TIER_NAMES, p_live)),
        'wait_today_tier': TIER_NAMES[int(s0[0])], 'wait_summary': summ, 'wait_eval_days': len(Rw),
        'wait_method': method, 'events_end': ev['end'].max() if len(ev) else pd.NaT,
        'history': hist, 'wait_backtest': Rw,
    }
