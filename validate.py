"""Frozen-model regional diagnosis and development-only level holdouts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
import welding as w


def stats(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return dict(n=len(error), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
                p95=float(np.quantile(np.abs(error), .95)), worst=float(np.abs(error).max()))


def analyze(run):
    bundle = w.load_bundle(run)
    data = pd.read_csv(Path(bundle['dataset']) / 'dataset.csv')
    out = Path(run) / 'validation'
    out.mkdir(exist_ok=True)
    regional, worst, capacity = [], [], []
    # Diagnose the frozen test set only; no refit or threshold tuning here.
    for order in (1, 2):
        test = data[(data.process_order == order) & (data.split == 'test')].copy()
        for target in w.TARGETS:
            item = bundle['models'][(order, target)]
            pred = np.maximum(0, item['model'].predict(test[bundle['features']]))
            result = test.copy()
            result['prediction'] = pred
            result['error'] = pred-test[target]
            result['absolute_error'] = result.error.abs()
            result['covered'] = result.absolute_error <= item['radius']
            for feature in w.INPUTS[:-1]:
                for level, group in result.groupby(feature):
                    regional.append(dict(order=order, target=target, feature=feature, level=level,
                        **stats(group[target], group.prediction), coverage=float(group.covered.mean()),
                        small_sample=len(group)<20))
            for _, row in result.nlargest(10, 'absolute_error').iterrows():
                worst.append(dict(order=order, target=target, **{k: row[k] for k in
                    ['batch_id','case_id',*w.INPUTS[:-1],'prediction','error','absolute_error']}, actual=row[target]))
            scale = 1000 if target.startswith('max_') else 1
            capacity.append(dict(order=order, target=target, unit='mm' if scale==1000 else 'dimensionless',
                observed_test_mae=float(result.absolute_error.mean()*scale),
                observed_test_p95=float(result.absolute_error.quantile(.95)*scale),
                calibrated_guard_band=float(item['radius']*scale),
                engineering_tolerance=None, acceptance='pending_engineering_requirements'))
    pd.DataFrame(regional).to_csv(out/'regional_errors.csv', index=False)
    pd.DataFrame(worst).to_csv(out/'worst_cases.csv', index=False)
    pd.DataFrame(capacity).to_csv(out/'error_capacity.csv', index=False)

    # All level-holdout development fits exclude the frozen final test partition.
    # Refit the already chosen family, not an additional search using test labels.
    development = data[data.split != 'test']
    rows, predictions = [], []
    for feature in w.INPUTS[:-1]:
        levels = sorted(data[feature].unique())
        for level in levels:
            mode = 'boundary_extrapolation' if level in (levels[0],levels[-1]) else 'interior_interpolation'
            for order in (1,2):
                scoped = development[development.process_order==order]
                selected = bundle['models'][(order, w.TARGETS[0])]['model']
                fitting_pool = development if 'process_order' in bundle['features'] else scoped
                train = fitting_pool[fitting_pool[feature]!=level]
                held = scoped[scoped[feature]==level]
                if len(train)<5 or len(held)<2:
                    continue
                for target in w.TARGETS:
                    model = clone(bundle['models'][(order,target)]['model'])
                    if 'process_order' in bundle['features']:
                        pooled = getattr(model.steps[0][1], 'pooled', False)
                        pool = development if pooled else scoped
                        train = pool[pool[feature]!=level]
                    model.fit(train[bundle['features']],train[target])
                    pred = np.maximum(0, model.predict(held[bundle['features']]))
                    rows.append(dict(feature=feature,level=level,mode=mode,order=order,target=target,
                        n_train=len(train), **stats(held[target],pred)))
                    for (_, sample), estimate in zip(held.iterrows(),pred):
                        predictions.append(dict(feature=feature,level=level,mode=mode,order=order,target=target,
                            batch_id=sample.batch_id,case_id=sample.case_id,actual=sample[target],prediction=estimate))
            print(f'Completed {feature}={level}', flush=True)
    pd.DataFrame(rows).to_csv(out/'level_holdout_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'level_holdout_predictions.csv',index=False)
    w.dump(out/'protocol.json',dict(run=str(Path(run).resolve()),
        regional='frozen final test; diagnostic only; do not tune against this report',
        level_holdout='train+calibration only; frozen test excluded; selected model families fixed; development stress tests, not independent nested model selection',
        tolerance='Engineering limits unset. Observed errors and conformal radii are model capability, not engineering tolerances.'))
    generate_report(out)
    return out


def generate_report(out):
    regional=pd.read_csv(out/'regional_errors.csv')
    held=pd.read_csv(out/'level_holdout_metrics.csv')
    lines=['# 模型区域误差与参数水平留出验证', '',
        '区域分析使用冻结测试集，仅用于诊断。水平留出使用训练＋校准数据，排除最终测试集；固定已选模型族，不是嵌套选模后的独立泛化证明。', '',
        '## 最差区域（按每个目标的区域MAE排序）', '',
        '小于20条的区域标记为小样本；单参数分组可能掩盖交互效应。', '']
    for target in w.TARGETS:
        scale=1000 if target.startswith('max_') else 1
        unit='mm' if scale==1000 else '无量纲'
        lines += [f'### {target}（误差单位：{unit}）','']
        for row in regional[regional.target==target].nlargest(3,'mae').itertuples():
            lines.append(f'- 顺序{row.order}，{row.feature}={row.level:g}：n={row.n}，MAE={row.mae*scale:.6g}，P95={row.p95*scale:.6g}，覆盖率={row.coverage:.1%}。')
        lines += ['', '水平留出最差情况：']
        for mode in ['interior_interpolation','boundary_extrapolation']:
            sub=held[(held.target==target)&(held['mode']==mode)]
            row=sub.loc[sub.mae.idxmax()]
            lines.append(f'- {mode}：顺序{int(row["order"])}，{row.feature}={row.level:g}，n={int(row.n)}，MAE={row.mae*scale:.6g}，最大误差={row.worst*scale:.6g}。')
        lines.append('')
    lines += ['## 工程容差状态','',
        '尚未获得图纸限值、功能要求、测量精度及网格收敛误差，工程验收保持待确认。error_capacity.csv列出了模型当前误差能力，不构成合格限值。',
        '建议由产品限值L、代理误差预算b_model、仿真离散误差预算b_sim及其他误差预算共同确定保守门槛：预测值+b_model+b_sim+b_other≤L。各预算需要一致单位与独立证据；不得由当前测试误差反推产品允许偏差。',
        '未来继续据这些诊断改进模型时，本测试集属于已查看的基准集；最终发布需新增未参与改进的独立ANSYS验证点。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    print(analyze(parser.parse_args().run))
