"""Experiment A: exact M5 protocol; also capture uncertainties for Experiment B."""
from pathlib import Path
import sys
import time
import warnings
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'99_provenance'))
from evidence_common import *

def execute_model(model,out,stem):
    """Fit all grouped folds once, saving predictions, kernels, warnings, and metrics."""
    df,X,y=targets();predictions=[];kernels=[];warn=[]
    for protocol,fold,train,test in folds(df):
        start=time.perf_counter();g=gp(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');g.fit(X[train],y[train]);mu,sd=g.predict(X[test],return_std=True)
        assert np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>=0).all()
        config=f'{model}_frozen'
        predictions.append(prediction_rows(df,mu,protocol,fold,train,test,model,config,sd))
        warn.extend(warning_rows(caught,model,protocol,fold,config))
        k=g.kernel_;V=np.linalg.solve(g.L_,k(X[test],g.X_train_).T)
        manual=(k.diag(X[test])-np.einsum('ij,ij->j',V,V))*g._y_train_std**2
        np.testing.assert_allclose(sd**2,manual,rtol=1e-8,atol=1e-10)
        kernels.append(dict(model_name=model,configuration_id=config,protocol=protocol,fold_id=fold,kernel_repr=str(k),
            constant_value=float(k.k1.k1.constant_value),ell_distance=float(k.k1.k2.length_scale[0]),
            ell_angle=float(k.k1.k2.length_scale[1]),noise_variance_normalized=float(k.k2.noise_level),
            kernel_theta_json=json.dumps(k.theta.tolist()),kernel_bounds_log_json=json.dumps(k.bounds.tolist()),
            response_train_mean_db=float(g._y_train_mean),response_train_std_db=float(g._y_train_std),
            log_marginal_likelihood=float(g.log_marginal_likelihood_value_),fit_seconds=time.perf_counter()-start,
            observation_noise_variance_db2=float(k.k2.noise_level*g._y_train_std**2),variance_formula_max_abs_error_db2=float(np.max(abs(sd**2-manual))),
            alpha=0.,normalize_y=True,optimizer='fmin_l_bfgs_b',n_restarts_optimizer=2,random_state=42))
        print(model,protocol,fold,'RMSE',metric_values(predictions[-1])['rmse_db'],flush=True)
    pred=pd.concat(predictions,ignore_index=True);fm,sm=tables(pred)
    write_csv(pred,out/f'{stem}_all_predictions.csv');write_csv(fm,out/f'{stem}_fold_metrics.csv')
    write_csv(sm,out/f'{stem}_grouped_summary.csv');write_csv(pd.DataFrame(kernels),out/f'{stem}_optimized_kernels.csv')
    write_json(warn,out/f'{stem}_warnings.json')
    if model=='M5':
        for protocol,file in [('LODO','m5_lodo_predictions.csv'),('LOAO','m5_loao_predictions.csv'),('HOLDOUT_50_80','m5_50_80_predictions.csv')]:write_csv(pred[pred.protocol==protocol],out/file)
        subset=[]
        h=pred[pred.protocol=='HOLDOUT_50_80']
        for distance in [.5,.8]:subset.append(dict(model_name=model,protocol='HOLDOUT_50_80',subset_distance_m=distance,**metric_values(h[np.isclose(h.distance_m,distance)])))
        write_csv(pd.DataFrame(subset),out/'m5_holdout_distance_metrics.csv')
    return pred,fm,sm

def main():
    """Execute the frozen M5 grouped protocol without outcome-dependent changes."""
    execute_model('M5',FINAL/'01_m5_grouped','m5')

if __name__=='__main__':main()
