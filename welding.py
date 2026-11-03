"""Versioned welding surrogate pipeline. Run python welding.py --help."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
INPUTS = ['current_A', 'voltage_V', 'ambient_C', 'speed_m_s', 'process_order']
TARGETS = ['elastic_eqv_strain_max', 'total_eqv_strain_max', 'plastic_eqv_strain_max',
           'max_displacement_m', 'max_ux_m', 'max_uy_m', 'max_uz_m']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_success(path):
    frame = pd.read_csv(path)
    required = ['case_id', *INPUTS, *TARGETS]
    if set(required) - set(frame):
        raise ValueError(f'Missing columns: {set(required)-set(frame)}')
    frame[INPUTS + TARGETS] = frame[INPUTS + TARGETS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame[INPUTS + TARGETS].to_numpy()).all():
        raise ValueError('Non-finite values found')
    if not frame.process_order.isin([1, 2]).all() or (frame[TARGETS] < 0).any().any():
        raise ValueError('Invalid order or negative magnitude')
    return frame


def prepare():
    frames, sources = [], []
    for path in sorted((ROOT / 'data/raw').glob('*/success.csv')):
        frame = read_success(path)
        frame['batch_id'] = path.parent.name
        frame['record_id'] = frame.batch_id + ':' + frame.case_id.astype(str)
        if frame.record_id.duplicated().any():
            raise ValueError(f'Duplicate case IDs in {path}')
        frames.append(frame)
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not frames:
        raise ValueError('No imported batches')
    all_data = pd.concat(frames, ignore_index=True)
    # Never silently average conflicting repeats; retain every source record separately.
    for key, group in all_data.groupby(INPUTS):
        if len(group) > 1 and not np.allclose(group[TARGETS], group[TARGETS].iloc[0], rtol=1e-6, atol=1e-12):
            raise ValueError(f'Conflicting repeated condition {key}; review batches before merging')
    frame = all_data.drop_duplicates(INPUTS).copy()
    frame['group_id'] = frame[INPUTS[:-1]].apply(lambda r: '|'.join(format(float(x), '.12g') for x in r), axis=1)
    def split(key):
        bucket = int(hashlib.sha256(('welding-v2-42|' + key).encode()).hexdigest()[:8], 16) % 100
        return 'train' if bucket < 60 else ('calibration' if bucket < 80 else 'test')
    frame['split'] = frame.group_id.map(split)
    version = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT / 'data/processed' / version
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / 'dataset.csv', index=False)
    all_data.to_csv(out / 'source_records.csv', index=False)
    failures = []
    for path in sorted((ROOT / 'data/raw').glob('*/failed.csv')):
        item = pd.read_csv(path)
        item['batch_id'] = path.parent.name
        failures.append(item)
    if failures:
        pd.concat(failures, ignore_index=True).to_csv(out / 'failed_records.csv', index=False)
    dump(out / 'manifest.json', {'version': version, 'sources': sources, 'source_rows': len(all_data),
         'unique_rows': len(frame), 'splits': frame.split.value_counts().to_dict(),
         'split_rule': 'sha256(welding-v2-42|I|U|ambient|speed), 60/20/20; order paired'})
    print(f'Dataset: {out}')
    return out


def candidates():
    return {
        'mean': DummyRegressor(),
        'quadratic_ridge': make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1)),
        'extra_trees': ExtraTreesRegressor(n_estimators=150, min_samples_leaf=1, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=180, max_depth=2, learning_rate=.05, random_state=42),
    }


def train():
    dataset = prepare()
    frame = pd.read_csv(dataset / 'dataset.csv')
    run = ROOT / 'artifacts/runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    models, metrics, selection, predictions = {}, [], [], []
    for order in (1, 2):
        scoped = frame[frame.process_order == order]
        tr = scoped[scoped.split == 'train']; cal = scoped[scoped.split == 'calibration']; test = scoped[scoped.split == 'test']
        if min(len(tr), len(cal), len(test)) < 5:
            raise ValueError('Insufficient samples in a partition')
        features = INPUTS[:-1]
        for target in TARGETS:
            scores = {}
            cv = GroupKFold(n_splits=min(5, tr.group_id.nunique()))
            for name, prototype in candidates().items():
                errors = []
                for a, b in cv.split(tr, groups=tr.group_id):
                    fitted = clone(prototype).fit(tr.iloc[a][features], tr.iloc[a][target])
                    errors.append(mean_absolute_error(tr.iloc[b][target], fitted.predict(tr.iloc[b][features])))
                scores[name] = float(np.mean(errors))
                selection.append({'order': order, 'target': target, 'model': name, 'cv_mae': scores[name]})
            best = min(scores, key=scores.get)
            model = candidates()[best].fit(tr[features], tr[target])
            residuals = np.abs(cal[target].to_numpy() - np.maximum(0, model.predict(cal[features])))
            rank = min(len(residuals), math.ceil((len(residuals) + 1) * .95))
            radius = float(np.sort(residuals)[rank-1])
            pred = np.maximum(0, model.predict(test[features]))
            error = np.abs(test[target].to_numpy() - pred)
            metrics.append({'order': order, 'target': target, 'model': best,
                'mae': float(error.mean()), 'rmse': float(np.sqrt(mean_squared_error(test[target], pred))),
                'r2': float(r2_score(test[target], pred)), 'p95_absolute_error': float(np.quantile(error,.95)),
                'max_absolute_error': float(error.max()), 'interval_radius': radius,
                'test_coverage': float(np.mean(error <= radius)), 'n_train': len(tr), 'n_cal': len(cal), 'n_test': len(test)})
            predictions.extend({'case_id': row.case_id, 'order': order, 'target': target, 'actual': actual, 'prediction': estimate}
                               for row, actual, estimate in zip(test.itertuples(), test[target], pred))
            models[(order, target)] = {'model': model, 'radius': radius}
        print(f'Finished order {order}', flush=True)
    pd.DataFrame(metrics).to_csv(run / 'metrics.csv', index=False)
    pd.DataFrame(selection).to_csv(run / 'cv_selection.csv', index=False)
    pd.DataFrame(predictions).to_csv(run / 'test_predictions.csv', index=False)
    bundle = {'models': models, 'features': INPUTS[:-1], 'targets': TARGETS, 'dataset': str(dataset),
              'bounds': {c: [float(frame[c].min()), float(frame[c].max())] for c in INPUTS[:-1]}}
    joblib.dump(bundle, run / 'bundle.joblib')
    dump(run / 'metadata.json', {'dataset': str(dataset), 'status': 'candidate; not engineering approved',
         'interval': '95% marginal split conformal per target/order; not simultaneous or optimizer guarantee',
         'sklearn': __import__('sklearn').__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    print(f'Candidate model: {run}')


def load_bundle(path):
    return joblib.load(Path(path) / 'bundle.joblib')


def predict(bundle, values):
    if set(values) != set(INPUTS):
        raise ValueError(f'Exactly these inputs required: {INPUTS}')
    if not all(np.isfinite(float(values[c])) for c in INPUTS) or values['process_order'] not in (1, 2):
        raise ValueError('Invalid inputs')
    outside = [c for c, (lo, hi) in bundle['bounds'].items() if not lo <= values[c] <= hi]
    if outside:
        raise ValueError(f'Outside trained bounds: {outside}')
    x = pd.DataFrame([values])[bundle['features']]
    result = {}
    for target in TARGETS:
        item = bundle['models'][(int(values['process_order']), target)]
        value = max(0., float(item['model'].predict(x)[0]))
        result[target] = {'prediction': value, 'lower': max(0., value-item['radius']), 'upper': value+item['radius']}
    return {'outputs': result, 'status': 'surrogate estimate; fixed simulation configuration',
            'units': 'strain dimensionless; displacement m; direction magnitudes have no sign'}


def retrieve(bundle, request):
    current = request['current']
    predict(bundle, current)  # shared validation
    limits = request['limits']
    if not limits or set(limits) - set(TARGETS):
        raise ValueError('limits must contain known targets')
    if any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Invalid target limit')
    data = pd.read_csv(Path(bundle['dataset']) / 'dataset.csv')
    for key, value in request.get('locked', {}).items():
        if key not in INPUTS:
            raise ValueError(f'Unknown locked parameter {key}')
        data = data[np.isclose(data[key], value, rtol=0, atol=1e-12)]
    for key, limit in limits.items():
        data = data[data[key] <= limit]
    cost = np.zeros(len(data))
    for key, (lo, hi) in bundle['bounds'].items():
        cost += np.abs(data[key].to_numpy()-current[key]) / (hi-lo if hi > lo else 1)
    cost += (data.process_order.to_numpy() != current['process_order']).astype(float)
    data = data.assign(adjustment_cost=cost).sort_values('adjustment_cost').head(5)
    return {'status': 'existing ANSYS cases only; not continuous optimization', 'count': len(data),
            'candidates': data[['batch_id', 'case_id', *INPUTS, *TARGETS, 'adjustment_cost']].to_dict('records')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('import-batch'); imp.add_argument('--batch', required=True); imp.add_argument('--success', required=True); imp.add_argument('--failed')
    sub.add_parser('prepare'); sub.add_parser('train')
    for cmd in ('predict', 'recommend'):
        p = sub.add_parser(cmd); p.add_argument('--run', required=True); p.add_argument('--input', required=True); p.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'import-batch':
        if not args.batch or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.batch):
            raise ValueError('Batch name must use letters, digits, underscore or hyphen')
        read_success(args.success)
        destination = ROOT / 'data/raw' / args.batch
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.success, destination / 'success.csv')
        if args.failed:
            shutil.copyfile(args.failed, destination / 'failed.csv')
        dump(destination / 'batch.json', {'batch': args.batch, 'settings': 'See docs/data_contract.md; verify compatibility before training'})
    elif args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train()
    else:
        bundle = load_bundle(args.run)
        request = json.loads(Path(args.input).read_text(encoding='utf-8-sig'))
        result = predict(bundle, request) if args.command == 'predict' else retrieve(bundle, request)
        if args.output:
            dump(Path(args.output), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
