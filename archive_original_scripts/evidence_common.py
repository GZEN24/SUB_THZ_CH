"""Round-3 evidence utilities. Original project modules are never imported.

Python 3.10; NumPy/SciPy/pandas/scikit-learn. All outputs stay in Final_paper.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[1]
RUN_ID = 'ojcoms_round3_20261009'
Z90 = 1.645  # historical convention, retained consistently
ALPHA_PI = 0.10
REPRO_TOL_DB = 0.001  # material reproduction discrepancy, declared before fits

def write_csv(df, path):
    """Exclusively create a full-precision CSV; never replace existing results."""
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        df.to_csv(stream, index=False, float_format='%.17g')

def write_json(obj, path):
    """Exclusively create a JSON artifact."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False, default=str)

def write_text(value, path):
    """Exclusively create a UTF-8 report."""
    Path(path).write_text(value, encoding='utf-8') if not Path(path).exists() else (_ for _ in ()).throw(FileExistsError(path))

def sha256(path):
    """Stream a file hash without loading large archives into memory."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def original_functions():
    """Compile only audited original function definitions; no module side effects."""
    env=dict(np=np, scipy=scipy, re=re, Path=Path, StandardScaler=StandardScaler,
             MLPRegressor=MLPRegressor, Pipeline=Pipeline, ANGLE_CAP_DEG=20,
             RANDOM_SEED=42, POLY2D_DEGREE=3)
    selected={
        'block2_complete.py':['_col_to_angle_map','load_karakoca_file','load_all_karakoca',
            'make_2d_features','fit_separable','sep_predict','_poly2d_phi','fit_poly2d','poly2d_predict','build_mlp'],
        'lodo_loao_cv.py':['fit_separable_theta0_held_out']}
    for filename,names in selected.items():
        tree=ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(nodes)==len(names)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(ROOT/filename),'exec'),env)
    return env

def targets():
    """Return verified row-indexed observations, deterministic features, response."""
    df=pd.read_csv(FINAL/'00_baseline/pl_targets_153.csv',float_precision='round_trip')
    X=np.column_stack([np.log10(df.distance_m.to_numpy()/.2),df.angle_deg.to_numpy()/20])
    return df,X,df.measured_pl_db.to_numpy()

def folds(df):
    """Yield exact grouped train/test memberships on distance-major indices."""
    for protocol,col in [('LODO','distance_m'),('LOAO','angle_deg')]:
        values=sorted(df[col].unique())
        for value in values:
            test=np.flatnonzero(np.isclose(df[col],value)); train=np.setdiff1d(np.arange(len(df)),test)
            label=f'{int(round(value*100))}cm' if col=='distance_m' else f'{int(value)}deg'
            yield protocol,label,train,test
    test=np.flatnonzero(np.isclose(df.distance_m,.5)|np.isclose(df.distance_m,.8))
    yield 'HOLDOUT_50_80','50_80cm',np.setdiff1d(np.arange(len(df)),test),test

def gp(model):
    """Construct exact frozen M4/M5 grouped fitting protocols."""
    k=RBF([.3,8.],[(.01,5.),(.5,100.)]) if model=='M4' else Matern([.3,8.],[(.01,5.),(.05,100.)],nu=2.5)
    return GaussianProcessRegressor(kernel=ConstantKernel(10.,(1.,1000.))*k+WhiteKernel(.1,(1e-4,10.)),
        alpha=0.,normalize_y=True,n_restarts_optimizer=2,random_state=42)

def prediction_rows(df,yhat,protocol,fold,train,test,model,config,std=None,extra=None):
    """Create auditable predictions; residual is prediction minus measured, in dB."""
    out=df.iloc[test][['row_id','distance_m','angle_deg','measured_pl_db']].copy()
    out['predicted_pl_db']=yhat
    out['residual_db']=out.predicted_pl_db-out.measured_pl_db
    out['squared_error_db2']=out.residual_db**2
    out['protocol']=protocol;out['fold_id']=fold;out['model_name']=model
    out['configuration_id']=config;out['train_count']=len(train);out['test_count']=len(test)
    out['run_id']=RUN_ID
    out['prediction_id']=[f'{RUN_ID}:{config}:{protocol}:{fold}:{r}' for r in out.row_id]
    if std is not None:
        out['predictive_std_db']=std;out['half_width_db']=Z90*std
        out['lower90_db']=out.predicted_pl_db-out.half_width_db
        out['upper90_db']=out.predicted_pl_db+out.half_width_db
        out['full_width_db']=2*out.half_width_db
        out['covered']=(out.measured_pl_db>=out.lower90_db)&(out.measured_pl_db<=out.upper90_db)
        out['interval_score_db']=out.full_width_db+20*np.maximum(out.lower90_db-out.measured_pl_db,0)+20*np.maximum(out.measured_pl_db-out.upper90_db,0)
    if extra:
        for k,v in extra.items():out[k]=v
    return out

def metric_values(df):
    """Metrics in dB; coverage is a fraction with explicit numerator/denominator."""
    r=df.residual_db.to_numpy()
    result=dict(n_points=len(df),rmse_db=float(np.sqrt(np.mean(r*r))),mae_db=float(np.mean(np.abs(r))))
    if 'covered' in df and df['covered'].notna().all():
        result.update(covered_count=int(df.covered.sum()),coverage_denominator=len(df),
            coverage=float(df.covered.mean()),mean_full_width_db=float(df.full_width_db.mean()),
            median_full_width_db=float(df.full_width_db.median()),mean_half_width_db=float(df.half_width_db.mean()),
            interval_score_db=float(df.interval_score_db.mean()))
    return result

def tables(pred):
    """Separate fold metrics, pooled points, and macro fold statistics."""
    keys=['model_name','configuration_id','protocol'];foldrows=[];summary=[]
    for key,g in pred.groupby(keys,sort=False):
        local=[]
        for fold,h in g.groupby('fold_id',sort=False):
            row=dict(zip(keys,key),fold_id=fold,**metric_values(h));local.append(row);foldrows.append(row)
        f=pd.DataFrame(local)
        summary.append(dict(zip(keys,key),aggregation='pooled_points',fold_count=len(f),**metric_values(g)))
        macro={k:float(f[k].mean()) for k in ['rmse_db','mae_db']}
        macro.update(median_fold_rmse_db=float(f.rmse_db.median()),sd_fold_rmse_db=float(f.rmse_db.std(ddof=1)) if len(f)>1 else 0.,
            min_fold_rmse_db=float(f.rmse_db.min()),max_fold_rmse_db=float(f.rmse_db.max()))
        if 'coverage' in f:
            for col in ['coverage','mean_full_width_db','median_full_width_db','mean_half_width_db','interval_score_db']:macro[col]=float(f[col].mean())
            macro['covered_count']=int(g.covered.sum());macro['coverage_denominator']=len(g)
        summary.append(dict(zip(keys,key),aggregation='macro_folds',fold_count=len(f),n_points=len(g),**macro))
    return pd.DataFrame(foldrows),pd.DataFrame(summary)

def warning_rows(caught,model,protocol,fold,config):
    """Preserve every captured warning with its fit context."""
    return [dict(model_name=model,protocol=protocol,fold_id=fold,configuration_id=config,
        category=w.category.__name__,message=str(w.message),filename=w.filename,lineno=w.lineno) for w in caught]
