"""Reproducible bounded candidate search with executable parameter steps."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import welding as w


def grid(bundle, request):
    current=request['current']
    w.predict(bundle,current)
    allowed=set(w.INPUTS)
    for key in ('locked','steps','max_change'):
        if set(request.get(key,{}))-allowed:
            raise ValueError(f'Unknown parameter in {key}')
    axes=[]
    for name in w.INPUTS[:-1]:
        if name not in bundle['bounds']:
            continue
        lo,hi=bundle['bounds'][name]
        change=request.get('max_change',{}).get(name,hi-lo)
        if not np.isfinite(change) or change<0:
            raise ValueError('max_change must be finite and nonnegative')
        lower,upper=max(lo,current[name]-change),min(hi,current[name]+change)
        step=request['steps'][name]
        if not np.isfinite(step) or step<=0:
            raise ValueError('steps must be positive and finite')
        # Lattice anchored at trained lower bound, never round after scoring.
        indices=np.arange(np.ceil((lower-lo)/step-1e-9),np.floor((upper-lo)/step+1e-9)+1)
        axis=lo+indices*step
        if name in request.get('locked',{}):
            value=float(request['locked'][name])
            if not lower-1e-12<=value<=upper+1e-12 or not np.isclose((value-lo)/step,round((value-lo)/step),atol=1e-8):
                raise ValueError(f'Locked value outside bounds/change/step: {name}')
            axis=np.array([value])
        axes.append(axis)
    orders=request.get('orders',[1,2])
    if 'process_order' in request.get('locked',{}):
        orders=[request['locked']['process_order']]
    if not orders or any(s not in (1,2) for s in orders):
        raise ValueError('orders must contain 1 and/or 2')
    count=int(np.prod([len(a) for a in axes]))*len(orders)
    if count>250000:
        raise ValueError(f'{count} candidates exceeds 250000; increase steps or restrict range')
    rows=[(*values,s) for values in itertools.product(*axes) for s in orders]
    return pd.DataFrame(rows,columns=[*bundle['bounds'],'process_order'])


def optimize(run,request,output):
    bundle=w.load_bundle(run)
    config=json.loads((w.ROOT/'configs/engineering_tolerances.json').read_text(encoding='utf-8'))
    limits={'max_displacement_m':config['residual_displacement_limit']/1000,
            'total_eqv_strain_max':config['total_strain_limit'],
            'plastic_eqv_strain_max':config['plastic_strain_limit']}
    budgets={t:config['model_error_budget_displacement']/1000 if t.startswith('max_')
             else config['model_error_budget_strain'] for t in limits}
    candidates=grid(bundle,request)
    total=len(candidates)
    feasible=np.ones(total,dtype=bool)
    for order in (1,2):
        idx=candidates.index[candidates.process_order==order]
        if not len(idx):
            continue
        x=candidates.loc[idx,bundle['features']]
        for target in limits:
            item=bundle['models'][(order,target)]
            pred=np.maximum(0,item['model'].predict(x))
            conservative=pred+max(budgets[target],item['radius'])
            candidates.loc[idx,target]=pred
            candidates.loc[idx,target+'_upper']=conservative
            feasible[idx]&=conservative<=limits[target]
    candidates['feasible']=feasible
    candidates['adjustment_cost']=0.
    for name,(lo,hi) in bundle['bounds'].items():
        candidates['adjustment_cost']+=abs(candidates[name]-request['current'][name])/(hi-lo or 1)
    candidates['adjustment_cost']+=(candidates.process_order!=request['current']['process_order']).astype(float)
    candidates['worst_limit_ratio']=np.maximum.reduce([candidates[t+'_upper'].to_numpy()/limits[t] for t in limits]) if total else []
    data=pd.read_csv(Path(bundle['dataset'])/'dataset.csv')
    keys=set(tuple(row) for row in data[w.INPUTS].round(10).to_numpy())
    candidates['existing_simulation']=[tuple(row) in keys for row in candidates[w.INPUTS].round(10).to_numpy()]
    candidates['verification_status']='pending_ANSYS; surrogate estimate'
    candidates['risk_note']=np.where(candidates.current_A>=225,'high-current region: elevated observed error','global interval does not guarantee local coverage')
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    candidates.to_csv(out/'all_candidates.csv',index=False)
    valid=candidates[candidates.feasible].sort_values(['adjustment_cost','worst_limit_ratio'])
    selected=valid.head(5)
    selected.to_csv(out/'recommendations.csv',index=False)
    # Explicitly export new points only for simulation, never relabel known points as new evidence.
    valid[~valid.existing_simulation].head(10).to_csv(out/'ansys_verification_queue.csv',index=False)
    if selected.empty:
        candidates.sort_values('worst_limit_ratio').head(5).to_csv(out/'closest_infeasible.csv',index=False)
    w.dump(out/'manifest.json',dict(run=str(Path(run).resolve()),request=request,tolerance_config=config,
        evaluated=total,feasible=len(valid),returned=len(selected),
        method='exhaustive executable lattice inside bounds; not continuous-global optimum',
        acceptance='prediction + max(project error budget, calibrated radius) <= limit',
        status='simulation surrogate development; ANSYS review required; no experimental qualification'))
    print(f'Evaluated {total}; feasible {len(valid)}; returned {len(selected)}; output {out}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    optimize(args.run,json.loads(Path(args.input).read_text(encoding='utf-8-sig')),args.output)
