"""R2.5: hold out 50 and 80 cm, reusing the published grouped-CV functions.

Python 3.10+, NumPy/pandas/scikit-learn; outputs never replace prior artifacts.
Run from the project root: python -B leave_two_interior_out.py
"""
from pathlib import Path
import inspect
import hashlib
import os
import json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'block2_output/round2_revision'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mplconfig'))
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import lodo_loao_cv as published
from round2_analysis import csv_new, json_new


def main() -> None:
    """Train on 119 original grid points; evaluate 34 held-out points."""
    names=['r2_5_leave_two_interior_summary.csv','r2_5_leave_two_interior_predictions.csv','r2_5_protocol.json']
    if any((OUT/x).exists() for x in names):raise FileExistsError('R2.5 evidence already exists; refusing replacement')
    np.random.seed(published.RANDOM_SEED)
    data=published.load_all_karakoca(ROOT/'Datasets/Sub_THz_measurements')
    d,a,pl,_,_=published.extract_statistics(data,angle_cap=published.ANGLE_CAP_DEG)
    held=np.isclose(d,.5)|np.isclose(d,.8)
    train_d,test_d=d[~held],d[held]
    Xtrain,_,_=published.make_2d_features(train_d,a)
    Xtest,dt,at=published.make_2d_features(test_d,a)
    ytrain,ytest=pl[~held].ravel(),pl[held].ravel()
    assert Xtrain.shape==(119,2) and Xtest.shape==(34,2)
    wlos,wang=published.fit_separable(train_d,a,pl[~held])
    pred={'M1':published.sep_predict(dt,at,wlos,wang)}
    w=published.fit_poly2d(Xtrain,ytrain);pred['M2']=published.poly2d_predict(Xtest,w)
    mlp=published.build_mlp();mlp.fit(Xtrain,ytrain);pred['M3']=mlp.predict(Xtest)
    captured=[]
    original=published.GaussianProcessRegressor
    def capture_gpr(*args,**kwargs):
        """Capture the regressor instantiated by the unchanged published wrapper."""
        g=original(*args,**kwargs);captured.append(g);return g
    published.GaussianProcessRegressor=capture_gpr
    try:pred['M4']=published.fit_predict_gpr(Xtrain,ytrain,Xtest)
    finally:published.GaussianProcessRegressor=original
    gpr=captured[0]
    indices=np.arange(len(d)*len(a)).reshape(len(d),len(a))
    details=pd.DataFrame({'original_index':indices[held].ravel(),'distance_m':dt,'angle_deg':at,'measured_pl_db':ytest})
    rows=[]
    for model,yp in pred.items():
        details[f'{model}_prediction_db']=yp;details[f'{model}_residual_db']=yp-ytest
        for label,mask in [('combined',np.ones(len(ytest),bool)),('50cm',np.isclose(dt,.5)),('80cm',np.isclose(dt,.8))]:
            err=yp[mask]-ytest[mask]
            rows.append(dict(protocol='leave_two_interior_out',subset=label,model=model,n_test=int(mask.sum()),
                             rmse_db=float(np.sqrt(np.mean(err**2))),mae_db=float(np.mean(np.abs(err)))))
    ref=pd.read_csv(OUT/'r2_2_cv_spatial_support.csv').set_index('model')
    for model in pred:
        for field in ['loo_rmse_db','lodo_avg_rmse_db']+[f'lodo_{cm}cm_rmse_db' for cm in [20,30,50,80,100]]:
            rows.append(dict(protocol='frozen_reference',subset=field,model=model,rmse_db=ref.loc[model,field]))
    csv_new(pd.DataFrame(rows),names[0]);csv_new(details,names[1])
    json_new(dict(training_indices_zero_based=indices[~held].ravel().tolist(),test_indices_zero_based=indices[held].ravel().tolist(),
        indexing='distance-major then sorted angle, original 9x17 grid',training_distances_m=train_d.tolist(),
        test_distances_m=test_d.tolist(),angles_deg=a.tolist(),seed=published.RANDOM_SEED,
        feature_definition='[log10(d/0.20), theta/20]; MLP StandardScaler fitted on training only',
        m4_kernel=str(gpr.kernel_),m4_kernel_theta=gpr.kernel_.theta.tolist(),
        m4_constructor={k:str(v) for k,v in gpr.get_params(deep=False).items()},
        grouped_source_sha256=hashlib.sha256((ROOT/'lodo_loao_cv.py').read_bytes()).hexdigest(),
        core_source_sha256=hashlib.sha256((ROOT/'block2_complete.py').read_bytes()).hexdigest(),
        reused_gpr_function=inspect.getsource(published.fit_predict_gpr),
        mlp_parameters={k:str(v) for k,v in mlp.get_params().items()},
        residual_convention='prediction minus measured',arbitrary_success_threshold=None),names[2])
    print(pd.DataFrame(rows[:12]).to_string(index=False));print('M4 kernel:',gpr.kernel_)


if __name__=='__main__':main()
