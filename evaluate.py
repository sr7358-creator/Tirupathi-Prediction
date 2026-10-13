"""Ablation study for the capstone report:  python evaluate.py
Walk-forward (expanding window, 30-day blocks) - every method is scored on the SAME days."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from utils import model_pipeline as mp

PATH = 'data/ttd_data.xlsx'
df, ev = mp.load_data(PATH)
X, Y, last = mp.build_features(df, ev)
pd.set_option('display.width', 200)

Rw = mp.evaluate_wait(X, Y, start='2024-07-01')
S = mp.summarize_wait(Rw).round(2)
S.to_csv('reports/ablation_wait.csv', index=False)
print(f"\nWAIT TIME - {len(Rw)} walk-forward days ({Rw.index.min().date()} to {Rw.index.max().date()})")
print(S.to_string(index=False))

rows = []
for col, name in [('crowd', 'Crowd'), ('tons', 'Tonsures')]:
    R = mp.evaluate_level(X, Y, col, start='2024-07-01', delta=True)
    t = mp.summarize_level(R).round(2).reset_index().rename(columns={'index': 'method'})
    t.insert(0, 'target', name); rows.append(t)
L = pd.concat(rows); L.to_csv('reports/ablation_crowd_tonsures.csv', index=False)
print("\nCROWD / TONSURES"); print(L.to_string(index=False))
