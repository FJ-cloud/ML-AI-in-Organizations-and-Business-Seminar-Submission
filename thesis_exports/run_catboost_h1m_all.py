#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

try:
	from catboost import CatBoostClassifier
	HAS_CAT = True
except Exception:
	HAS_CAT = False

SCRIPT_DIR = Path(__file__).resolve().parent

# Output dir: prefer repo export/, else local outputs/
def _detect_repo_root(start: Path) -> Path | None:
	for p in [start, *start.parents]:
		if (p / 'data').exists() and (p / 'export').exists():
			return p
	return None

_REPO_ROOT = _detect_repo_root(SCRIPT_DIR)
if _REPO_ROOT is not None:
	OUT_DIR = _REPO_ROOT / 'export'
else:
	OUT_DIR = SCRIPT_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _candidate_model_ready_dirs() -> list[Path]:
	roots: list[Path] = []
	for p in [SCRIPT_DIR, Path.cwd(), *SCRIPT_DIR.parents]:
		d = p / 'data' / 'processed' / 'model_ready'
		if d not in roots:
			roots.append(d)
	return roots


def _latest_in_dir(directory: Path, pattern: str) -> Path | None:
	files = sorted(directory.glob(pattern))
	return files[-1] if files else None


def latest_model_lag1() -> Path:
	# Prefer CSV next to the script
	local = _latest_in_dir(SCRIPT_DIR, 'model_lag1_*.csv')
	if local is not None:
		return local
	# Fallback to repo-style layout
	candidates: list[Path] = []
	for d in _candidate_model_ready_dirs():
		if d.exists():
			candidates.extend(sorted(d.glob('model_lag1_*.csv')))
	if not candidates:
		raise SystemExit('No model_lag1_*.csv found. Place it next to this script or under data/processed/model_ready.')
	return sorted(candidates)[-1]


@dataclass
class Config:
	split_date: str = '2022-08-31'
	roll_windows: tuple[int, ...] = (3, 6)
	feature_freq_threshold: float = 0.6
	target_col: str = 'will_fail_within_1m'
	drop_post_failure: bool = True
	save_preds: bool = True
	use_catboost: bool = True


def canonicalize_macros(df: pd.DataFrame) -> pd.DataFrame:
	# Drop macro variants; prefer canonical names where both exist
	drop_cols = []
	prefer_pairs = [
		('ukraine_3y_yield', 'ukraine_3y_yield_z'),
		('us_3y_yield', 'us_3y_yield_z'),
		('yield_spread_ukr_us_3y', 'yield_spread_ukr_us_3y_z'),
	]
	for base, zname in prefer_pairs:
		if zname in df.columns and base in df.columns:
			drop_cols.append(zname)
	# Also drop duplicate lag naming artifacts if present
	drop_cols += [c for c in df.columns if c.endswith('_lag1.1')]
	return df.drop(columns=[c for c in drop_cols if c in df.columns])


def add_trend_volatility(df: pd.DataFrame, feat_cols: list[str], windows=(3, 6)) -> tuple[pd.DataFrame, list[str]]:
	df = df.sort_values(['bank_name', 'date_m']).copy()
	new_cols: list[str] = []
	for w in windows:
		for c in feat_cols:
			col_ma = f'{c}_ma{w}'
			col_std = f'{c}_std{w}'
			df[col_ma] = df.groupby('bank_name')[c].transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).mean())
			df[col_std] = df.groupby('bank_name')[c].transform(lambda s: s.rolling(w, min_periods=max(1, w // 2)).std())
			new_cols.extend([col_ma, col_std])
	return df, new_cols


def time_split(df: pd.DataFrame, split_date: str):
	dt = pd.Timestamp(split_date)
	return df[df['date_m'] <= dt].copy(), df[df['date_m'] > dt].copy()


def get_features(df: pd.DataFrame) -> list[str]:
	non_features = {'bank_name', 'date_m', 'year_month', 'failed', 'will_fail_within_6m', 'will_fail_within_1m', 'post_failure', 'processing_timestamp', 'failure_date', 'last_reporting_date'}
	feat_cols = [c for c in df.columns if c not in non_features]
	feat_cols = [c for c in feat_cols if not c.endswith('_present')]
	return feat_cols


def stability_selection(train_df: pd.DataFrame, feat_cols: list[str], y_col: str, n_splits: int = 5, threshold: float = 0.6) -> tuple[list[str], dict[str, int]]:
	# Rolling time splits: equal partitions over time
	dates = np.sort(train_df['date_m'].unique())
	split_points = np.linspace(0, len(dates), n_splits + 1, dtype=int)
	counts = {c: 0 for c in feat_cols}
	valid_slices = 0

	for i in range(n_splits):
		start, end = split_points[i], split_points[i + 1]
		if end - start < 3:
			continue
		cut = dates[start:end]
		tr = train_df[train_df['date_m'].isin(cut)]
		y = tr[y_col].astype(int)
		if y.nunique() < 2:
			continue
		valid_slices += 1
		# Simple L1 logistic for selection
		X = tr[feat_cols].astype(float).values
		y = y.values
		imputer = SimpleImputer(strategy='median')
		X = imputer.fit_transform(X)
		scaler = StandardScaler(with_mean=False)
		Xs = scaler.fit_transform(X)
		model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=2000)
		model.fit(Xs, y)
		selected = [feat_cols[j] for j, w in enumerate(model.coef_.ravel()) if abs(w) > 1e-6]
		for c in selected:
			counts[c] += 1

	denom = max(1, valid_slices)
	keep = [c for c in feat_cols if counts[c] / denom >= threshold]
	return (keep or feat_cols), counts


def train_model(X_tr, y_tr, X_te, use_catboost=True):
	print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting model fit (catboost={use_catboost and HAS_CAT})...")
	start = time.time()
	if HAS_CAT and use_catboost:
		model = CatBoostClassifier(
			loss_function='Logloss',
			eval_metric='AUC',
			depth=6,
			learning_rate=0.03,
			l2_leaf_reg=6.0,
			iterations=2000,
			auto_class_weights='Balanced',
			random_strength=1.0,
			border_count=128,
			verbose=False,
			random_seed=42,
		)
		model.fit(X_tr, y_tr)
		p_tr = model.predict_proba(X_tr)[:, 1]
		p_te = model.predict_proba(X_te)[:, 1]
		print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished model fit in {time.time()-start:.1f}s")
		return model, p_tr, p_te
	# Logistic fallback
	imputer = SimpleImputer(strategy='median')
	X_tr = imputer.fit_transform(X_tr)
	X_te = imputer.transform(X_te)
	scaler = StandardScaler(with_mean=False)
	Xs_tr = scaler.fit_transform(X_tr)
	Xs_te = scaler.transform(X_te)
	model = LogisticRegression(max_iter=5000, class_weight='balanced')
	model.fit(Xs_tr, y_tr)
	print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished logistic fit in {time.time()-start:.1f}s")
	return model, model.predict_proba(Xs_tr)[:, 1], model.predict_proba(Xs_te)[:, 1]


def score(y_tr, p_tr, y_te, p_te):
	def s(y, p):
		return dict(
			roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
			pr_auc=float(average_precision_score(y, p)),
			brier=float(brier_score_loss(y, p)),
			n=int(len(y)),
		)

	return {'train': s(y_tr, p_tr), 'test': s(y_te, p_te)}


def run(cfg: Config, csv: str | None, top_k: int = 0, use_all: bool = True):
	df = pd.read_csv(csv or latest_model_lag1(), parse_dates=['date_m'])
	df = canonicalize_macros(df)

	# Choose target
	y_col = cfg.target_col if cfg.target_col in df.columns else 'failed'

	# Drop post-failure months for early-warning
	if cfg.drop_post_failure and 'post_failure' in df.columns:
		df = df[df['post_failure'] == False].copy()

	feat_cols = get_features(df)
	df, new_cols = add_trend_volatility(df, feat_cols, windows=cfg.roll_windows)
	feat_cols = feat_cols + new_cols

	df_tr, df_te = time_split(df, cfg.split_date)

	# Stability selection (unless using all features)
	if use_all:
		selected_feats = feat_cols
		stability_counts = {c: 0 for c in feat_cols}
	else:
		stable_feats, counts = stability_selection(df_tr, feat_cols, y_col, n_splits=5, threshold=cfg.feature_freq_threshold)
		ranked = sorted(stable_feats, key=lambda c: (-counts.get(c, 0), c))
		selected_feats = ranked[:top_k] if top_k and top_k > 0 else ranked
		stability_counts = counts

	# Train model
	X_tr = df_tr[selected_feats].astype(float).values
	y_tr = df_tr[y_col].astype(int).values
	X_te = df_te[selected_feats].astype(float).values
	y_te = df_te[y_col].astype(int).values

	model, p_tr, p_te = train_model(X_tr, y_tr, X_te, use_catboost=cfg.use_catboost)
	metrics = score(y_tr, p_tr, y_te, p_te)

	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	payload = {
		'config': cfg.__dict__,
		'csv': str(csv or latest_model_lag1()),
		'split_date': cfg.split_date,
		'selected_features_count': len(selected_feats),
		'selected_features': selected_feats[:100],
		'stability_counts_top': dict(sorted(stability_counts.items(), key=lambda kv: -kv[1])[:50]),
		'target_col': y_col,
		'dropped_post_failure': bool(cfg.drop_post_failure and 'post_failure' in df.columns),
		'metrics': metrics,
	}
	out = OUT_DIR / f'results_{ts}.json'
	pd.Series(payload).to_json(out, indent=2)

	# Save per-row predictions
	if cfg.save_preds:
		preds_tr = df_tr[['bank_name', 'date_m', 'year_month']].copy()
		preds_tr['split'] = 'train'
		preds_tr['y'] = y_tr
		preds_tr['p'] = p_tr
		preds_te = df_te[['bank_name', 'date_m', 'year_month']].copy()
		preds_te['split'] = 'test'
		preds_te['y'] = y_te
		preds_te['p'] = p_te
		preds = pd.concat([preds_tr, preds_te], axis=0, ignore_index=True)
		preds_out = OUT_DIR / f'predictions_{ts}.csv'
		preds.sort_values(['split', 'date_m', 'bank_name']).to_csv(preds_out, index=False)
		print('Predictions saved:', preds_out)

	print('Saved:', out)
	print('Target:', y_col, '| Dropped post-failure:', bool(cfg.drop_post_failure and 'post_failure' in df.columns))
	print('Test ROC=%.3f PR=%.3f Brier=%.4f (n=%d)' % (
		metrics['test']['roc_auc'], metrics['test']['pr_auc'], metrics['test']['brier'], metrics['test']['n']))


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument('--csv', type=str, default=None, help='Path to model-ready CSV (defaults to latest model_lag1_*.csv)')
	ap.add_argument('--split_date', type=str, default='2022-08-31')
	ap.add_argument('--no_cat', action='store_true')
	return ap.parse_args()


def main():
	args = parse_args()
	cfg = Config(split_date=args.split_date, use_catboost=not args.no_cat)
	run(cfg, args.csv, top_k=0, use_all=True)


if __name__ == '__main__':
	main() 