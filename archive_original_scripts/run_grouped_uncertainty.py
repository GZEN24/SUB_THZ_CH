"""Experiment B: refit M4, reuse the exact M5 grouped fits from Experiment A."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'99_provenance'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_m5_grouped'))
from evidence_common import *
from run_m5_grouped import execute_model

def main():
    """Evaluate nominal observation prediction intervals and frozen M4 agreement."""
    out=FINAL/'02_grouped_uncertainty'
    m4,fm4,sm4=execute_model('M4',out,'m4')
    m5=pd.read_csv(FINAL/'01_m5_grouped/m5_all_predictions.csv',float_precision='round_trip')
    comparisons=[]
    for protocol,filename,col in [('LODO','lodo_per_fold.csv','distance_cm'),('LOAO','loao_per_fold.csv','angle_deg')]:
        saved=pd.read_csv(ROOT/'block2_output'/filename,float_precision='round_trip')
        for _,row in saved.iterrows():
            fold=f'{int(round(row[col]))}cm' if protocol=='LODO' else f'{int(row[col])}deg'
            current=float(fm4[(fm4.protocol==protocol)&(fm4.fold_id==fold)].rmse_db.iloc[0])
            comparisons.append(dict(model_name='M4',protocol=protocol,fold_id=fold,frozen_rmse_db=row.M4,reproduced_rmse_db=current,
                absolute_difference_db=abs(current-row.M4),material_tolerance_db=REPRO_TOL_DB,pass_comparability=abs(current-row.M4)<=REPRO_TOL_DB))
    hold=pd.read_csv(ROOT/'block2_output/round2_revision/r2_5_leave_two_interior_summary.csv',float_precision='round_trip')
    ref=float(hold[(hold.model=='M4')&(hold.subset=='combined')].rmse_db.iloc[0]);cur=float(sm4[(sm4.protocol=='HOLDOUT_50_80')&(sm4.aggregation=='pooled_points')].rmse_db.iloc[0])
    comparisons.append(dict(model_name='M4',protocol='HOLDOUT_50_80',fold_id='50_80cm',frozen_rmse_db=ref,reproduced_rmse_db=cur,
        absolute_difference_db=abs(cur-ref),material_tolerance_db=REPRO_TOL_DB,pass_comparability=abs(cur-ref)<=REPRO_TOL_DB))
    write_csv(pd.DataFrame(comparisons),out/'m4_historical_reproduction.csv')
    for model,p in [('m4',m4),('m5',m5)]:write_csv(p,out/f'{model}_grouped_pi_predictions.csv')
    allpred=pd.concat([m4,m5],ignore_index=True);fm,sm=tables(allpred)
    write_csv(fm,out/'grouped_pi_fold_metrics.csv');write_csv(sm,out/'grouped_pi_summary.csv')
    write_csv(allpred[['prediction_id','model_name','configuration_id','protocol','fold_id','row_id','distance_m','angle_deg',
        'measured_pl_db','lower90_db','upper90_db','full_width_db','interval_score_db']],out/'grouped_pi_interval_scores.csv')
    if not all(x['pass_comparability'] for x in comparisons):
        raise RuntimeError('M4 historical comparability failure: see raw reproduction table; assessment must remain blocked.')
    print('M4 historical reproduction PASS',flush=True)

if __name__=='__main__':main()
