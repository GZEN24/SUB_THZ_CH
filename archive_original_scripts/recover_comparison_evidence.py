"""Recover available historical raw predictions; refit only cheap polynomial groups."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'99_provenance'))
from evidence_common import *

def main():
    """Supply raw comparison evidence without rerunning historical LOO models."""
    out=FINAL/'00_baseline';df,X,y=targets();f=original_functions();allpred=[]
    loo=pd.read_csv(ROOT/'block2_output/round2_revision/figure_loo_predictions.csv',float_precision='round_trip')
    m5=pd.read_csv(ROOT/'block2_output/m5_matern_loo_predictions.csv',float_precision='round_trip')
    for model in ['M1','M2','M3','M4','M5']:
        for idx in range(len(df)):
            train=np.setdiff1d(np.arange(len(df)),[idx]);mu=m5.y_pred_m5.iloc[idx] if model=='M5' else loo[f'{model}_prediction_db'].iloc[idx]
            sd=[m5.y_std_m5.iloc[idx]] if model=='M5' else [loo.M4_std_db.iloc[idx]] if model=='M4' else None
            p=prediction_rows(df,[mu],'LOO',f'row{idx}',train,np.array([idx]),model,f'{model}_historical_LOO',np.asarray(sd) if sd is not None else None)
            p['evidence_origin']='recovered_archived_raw';allpred.append(p)
    held=pd.read_csv(ROOT/'block2_output/round2_revision/r2_5_leave_two_interior_predictions.csv',float_precision='round_trip')
    test=held.original_index.to_numpy(dtype=int);train=np.setdiff1d(np.arange(153),test)
    for model in ['M1','M2','M3','M4']:
        p=prediction_rows(df,held[f'{model}_prediction_db'].to_numpy(),'HOLDOUT_50_80','50_80cm',train,test,model,f'{model}_historical_holdout')
        p['evidence_origin']='recovered_archived_raw';allpred.append(p)
    ext=pd.read_csv(ROOT/'block2_output/round2_revision/external_validation_predictions.csv',float_precision='round_trip')
    external=[]
    for model in ['M1','M2','M3','M4']:
        p=ext[['distance_m','measured_pl_db']].copy();p['angle_deg']=0;p['row_id']=[f'EXT{i}' for i in range(len(p))]
        p['predicted_pl_db']=ext[f'{model}_prediction_db'];p['residual_db']=p.predicted_pl_db-p.measured_pl_db;p['squared_error_db2']=p.residual_db**2
        p['model_name']=model;p['configuration_id']=f'{model}_historical_external';p['protocol']='EXTERNAL_TEKBIYIK';p['fold_id']='external_campaign'
        p['train_count']=153;p['test_count']=5;p['run_id']=RUN_ID;p['evidence_origin']='recovered_archived_raw'
        p['prediction_id']=[f'{RUN_ID}:{model}:EXTERNAL:{r}' for r in p.row_id];external.append(p)
    write_csv(pd.concat(external,ignore_index=True),out/'historical_external_predictions.csv')
    poly=[];repro=[]
    for protocol,fold,train,test in folds(df):
        if protocol=='HOLDOUT_50_80':continue # already have raw historical values
        for model in ['M1','M2']:
            if model=='M2':mu=f['poly2d_predict'](X[test],f['fit_poly2d'](X[train],y[train]))
            else:
                d=np.sort(df.iloc[train].distance_m.unique());a=np.sort(df.iloc[train].angle_deg.unique());matrix=df.iloc[train].measured_pl_db.to_numpy().reshape(len(d),len(a))
                fit=f['fit_separable_theta0_held_out'] if protocol=='LOAO' and fold=='0deg' else f['fit_separable']
                wlos,wang=fit(d,a,matrix);mu=f['sep_predict'](df.iloc[test].distance_m.to_numpy(),df.iloc[test].angle_deg.to_numpy(),wlos,wang)
            p=prediction_rows(df,mu,protocol,fold,train,test,model,f'{model}_grouped_reproduction');p['evidence_origin']='new_original_protocol_reproduction';poly.append(p)
            frozen=pd.read_csv(ROOT/'block2_output'/('lodo_per_fold.csv' if protocol=='LODO' else 'loao_per_fold.csv'),float_precision='round_trip')
            val=int(fold.removesuffix('cm').removesuffix('deg'));col='distance_cm' if protocol=='LODO' else 'angle_deg'
            ref=float(frozen.loc[np.isclose(frozen[col],val),model].iloc[0]);current=metric_values(p)['rmse_db']
            repro.append(dict(model_name=model,protocol=protocol,fold_id=fold,frozen_rmse_db=ref,reproduced_rmse_db=current,absolute_difference_db=abs(ref-current),pass_comparability=abs(ref-current)<=REPRO_TOL_DB))
    write_csv(pd.DataFrame(repro),out/'polynomial_grouped_reproduction.csv')
    assert all(r['pass_comparability'] for r in repro)
    raw=pd.concat(allpred+poly,ignore_index=True);write_csv(raw,out/'recovered_comparison_predictions.csv')
    fm,sm=tables(raw);write_csv(fm,out/'recovered_fold_metrics.csv');write_csv(sm,out/'recovered_summary.csv')
    ef,es=tables(pd.concat(external,ignore_index=True));write_csv(es,out/'historical_external_summary.csv')
    print('Recovered pointwise/holdout/external raw evidence and reproduced polynomial groups PASS.',flush=True)

if __name__=='__main__':main()
