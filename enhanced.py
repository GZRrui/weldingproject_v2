"""Enhanced surrogate comparison; frozen calibration and historical test retained."""
from pathlib import Path
from datetime import datetime
import argparse
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import welding as w
from validate import stats


class Features(TransformerMixin,BaseEstimator):
    def __init__(self,physical=False,pooled=False):
        self.physical=physical
        self.pooled=pooled

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        result=X[w.INPUTS[:-1]].to_numpy(dtype=float)
        if self.physical:
            power=X.current_A.to_numpy()*X.voltage_V.to_numpy()
            result=np.column_stack([result,power,power/X.speed_m_s.to_numpy()])
        if self.pooled:
            result=np.column_stack([result,(X.process_order.to_numpy()==2).astype(float)])
        return result


def options():
    base=w.candidates()
    base.update({
        'xgb_depth2':TransformedTargetRegressor(regressor=XGBRegressor(n_estimators=250,max_depth=2,learning_rate=.04,n_jobs=2,random_state=42,tree_method='hist'),transformer=StandardScaler()),
        'xgb_depth4':TransformedTargetRegressor(regressor=XGBRegressor(n_estimators=250,max_depth=4,learning_rate=.04,n_jobs=2,random_state=42,tree_method='hist'),transformer=StandardScaler()),
        # Fixed kernels form an explicit small hyperparameter grid, selected only by CV.
        **{f'gpr_matern_l{length}':make_pipeline(StandardScaler(),GaussianProcessRegressor(
            kernel=Matern(length_scale=length,nu=2.5),optimizer=None,alpha=1e-5,normalize_y=True)) for length in (.7,1.5,3.)}
    })
    return base


def train(oldrun):
    old=w.load_bundle(oldrun)
    data=pd.read_csv(Path(old['dataset'])/'dataset.csv')
    tr=data[data.split=='train']; cal=data[data.split=='calibration']; test=data[data.split=='test']
    out=w.ROOT/'artifacts/runs'/('enhanced_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    models={}; selection=[]; comparison=[]; predictions=[]
    for target in w.TARGETS:
        scores=[]
        for pooled in (False,True):
            for physical in (False,True):
                for name,base in options().items():
                    errors={1:[],2:[]}
                    pipeline=make_pipeline(Features(physical,pooled),base)
                    for a,b in GroupKFold(3).split(tr,groups=tr.group_id):
                        fit,held=tr.iloc[a],tr.iloc[b]
                        if pooled:
                            model=clone(pipeline).fit(fit,fit[target])
                            for order in (1,2):
                                group=held[held.process_order==order]
                                errors[order].extend(np.abs(np.maximum(0,model.predict(group))-group[target]))
                        else:
                            for order in (1,2):
                                group=held[held.process_order==order]
                                fit_order=fit[fit.process_order==order]
                                model=clone(pipeline).fit(fit_order,fit_order[target])
                                errors[order].extend(np.abs(np.maximum(0,model.predict(group))-group[target]))
                    row=dict(target=target,model=name,pooled=pooled,physical=physical,
                        cv_mae_order1=float(np.mean(errors[1])),cv_mae_order2=float(np.mean(errors[2])))
                    row['cv_balanced_mae']=(row['cv_mae_order1']+row['cv_mae_order2'])/2
                    scores.append((row,pipeline)); selection.append(row)
        best,pipeline=min(scores,key=lambda pair:pair[0]['cv_balanced_mae'])
        for order in (1,2):
            trainset=tr if best['pooled'] else tr[tr.process_order==order]
            model=clone(pipeline).fit(trainset,trainset[target])
            calibration=cal[cal.process_order==order]; held=test[test.process_order==order]
            residual=np.abs(np.maximum(0,model.predict(calibration))-calibration[target])
            radius=float(np.sort(residual)[min(len(residual),math.ceil((len(residual)+1)*.95))-1])
            pred=np.maximum(0,model.predict(held))
            baseline=np.maximum(0,old['models'][(order,target)]['model'].predict(held[old['features']]))
            row=dict(order=order,target=target,selected_model=best['model'],pooled=best['pooled'],physical=best['physical'],
                **stats(held[target],pred),r2=float(r2_score(held[target],pred)),baseline_mae=float(np.abs(baseline-held[target]).mean()),
                interval_radius=radius,coverage=float(np.mean(np.abs(pred-held[target])<=radius)))
            row['mae_improvement_fraction']=1-row['mae']/row['baseline_mae']
            comparison.append(row)
            models[(order,target)]={'model':model,'radius':radius}
            for (_,sample),p in zip(held.iterrows(),pred):
                predictions.append(dict(batch_id=sample.batch_id,case_id=sample.case_id,order=order,target=target,actual=sample[target],prediction=p,**{k:sample[k] for k in w.INPUTS[:-1]}))
        pd.DataFrame(selection).to_csv(out/'cv_selection.csv',index=False)
        print(f'{target}: {best}',flush=True)
    # Pipelines accept the full named input; transform selects raw/physical/order features.
    bundle=dict(models=models,features=w.INPUTS,targets=w.TARGETS,dataset=old['dataset'],bounds=old['bounds'])
    joblib.dump(bundle,out/'bundle.joblib')
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'test_predictions.csv',index=False)
    w.dump(out/'metadata.json',dict(baseline=str(oldrun),status='candidate, historical test already inspected',
        selection='3-fold group CV on original training partition only; 36 configurations per target',
        calibration='original independent calibration partition; 95% marginal intervals',
        gpr='fixed Matern nu=2.5; length .7/1.5/3 selected by CV; no marginal-likelihood optimization',
        dependencies={name:__import__(name).__version__ for name in ['numpy','pandas','sklearn','xgboost']}))
    report(out)
    print(out,flush=True)


def report(out):
    data=pd.read_csv(out/'comparison.csv')
    lines=['# 增强模型对比报告','','以原训练分区三折分组交叉验证选型；校准分区只估计区间；历史测试集只做对比，不能视为全新独立证据。','',
           '|顺序|目标|模型|统一建模|物理特征|旧MAE|新MAE|改善比例|新R²|','|---|---|---|---|---|---|---|---|---|']
    for r in data.itertuples():
        lines.append(f'|{r.order}|{r.target}|{r.selected_model}|{r.pooled}|{r.physical}|{r.baseline_mae:.6g}|{r.mae:.6g}|{r.mae_improvement_fraction:.1%}|{r.r2:.4f}|')
    lines+=['','MAE单位为原始数据单位（位移m，应变无量纲）。模型按开发CV选择，历史测试变差的输出如实保留，不按测试结果反选模型。旧版保留，当前不自动发布。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    # Stable module path for joblib, including execution as a script.
    from enhanced import train
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',required=True)
    train(parser.parse_args().baseline)
