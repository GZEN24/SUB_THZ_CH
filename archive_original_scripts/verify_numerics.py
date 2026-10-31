"""Independent reaggregation from saved observations; no fitting modules imported."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
FINAL=Path(__file__).resolve().parents[1]
KEYS=['model_name','configuration_id','protocol']
TESTS=[]

def check(name,condition,detail=''):
    """Retain a PASS/FAIL assertion without concealing later independent checks."""
    TESTS.append(dict(test=name,status='PASS' if bool(condition) else 'FAIL',detail=detail))

def close(a,b):
    """Tight CSV numerical equality, allowing irrelevant missing fields."""
    return bool(np.allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True))

def read(path):
    """Read saved full precision data."""
    return pd.read_csv(FINAL/path,float_precision='round_trip')

def calculated(g):
    """Recompute errors and proper interval metrics from y, mu, sigma alone."""
    residual=g.predicted_pl_db.to_numpy()-g.measured_pl_db.to_numpy()
    values=dict(n_points=len(g),rmse_db=float(np.linalg.norm(residual)/np.sqrt(len(g))),mae_db=float(np.abs(residual).sum()/len(g)))
    if 'predictive_std_db' in g and g.predictive_std_db.notna().all():
        sd=g.predictive_std_db.to_numpy();h=1.645*sd;lo=g.predicted_pl_db.to_numpy()-h;hi=g.predicted_pl_db.to_numpy()+h
        yt=g.measured_pl_db.to_numpy();cov=(yt>=lo)&(yt<=hi);width=hi-lo
        score=width.copy();below=yt<lo;above=yt>hi;score[below]+=20*(lo[below]-yt[below]);score[above]+=20*(yt[above]-hi[above])
        values.update(covered_count=int(cov.sum()),coverage_denominator=len(g),coverage=float(cov.sum()/len(g)),
            mean_full_width_db=float(width.mean()),median_full_width_db=float(np.median(width)),mean_half_width_db=float(h.mean()),interval_score_db=float(score.mean()))
    return values

def independent_tables(pred):
    """Rebuild complete fold/pooled/macro metric tables using independent formulas."""
    foldrows=[];summary=[]
    for key,g in pred.groupby(KEYS,sort=False):
        fold=[]
        for fold_id,h in g.groupby('fold_id',sort=False):
            row=dict(zip(KEYS,key),fold_id=fold_id,**calculated(h));fold.append(row);foldrows.append(row)
        f=pd.DataFrame(fold);pooled=calculated(g)
        summary.append(dict(zip(KEYS,key),aggregation='pooled_points',fold_count=len(f),**pooled))
        macro={c:float(f[c].sum()/len(f)) for c in ['rmse_db','mae_db']}
        macro.update(median_fold_rmse_db=float(np.median(f.rmse_db)),sd_fold_rmse_db=float(np.std(f.rmse_db,ddof=1)) if len(f)>1 else 0.,min_fold_rmse_db=float(min(f.rmse_db)),max_fold_rmse_db=float(max(f.rmse_db)))
        if 'coverage' in f:
            for c in ['coverage','mean_full_width_db','median_full_width_db','mean_half_width_db','interval_score_db']:macro[c]=float(f[c].sum()/len(f))
            macro.update(covered_count=pooled['covered_count'],coverage_denominator=pooled['coverage_denominator'])
        summary.append(dict(zip(KEYS,key),aggregation='macro_folds',fold_count=len(f),n_points=len(g),**macro))
    return pd.DataFrame(foldrows),pd.DataFrame(summary)

def compare(expected,saved,keys,name):
    """Compare every scientific metric supplied in a saved fold/summary table."""
    joined=saved.merge(expected,on=keys,how='outer',suffixes=('_saved','_calc'),indicator=True)
    check(name+' row membership',len(joined)==len(saved)==len(expected) and (joined._merge=='both').all())
    for c in expected.select_dtypes(include='number').columns:
        if c in saved:check(name+' '+c,close(joined[c+'_saved'],joined[c+'_calc']))

def verify_predictions(path,manifest,base):
    """Assert units, finite data, target identity, row counts, and test-group integrity."""
    pred=read(path);name=str(path)
    check(name+' unique prediction IDs',not pred.prediction_id.duplicated().any())
    check(name+' finite observations/predictions',np.isfinite(pred[['measured_pl_db','predicted_pl_db']]).all().all())
    check(name+' stored residual dB',close(pred.residual_db,pred.predicted_pl_db-pred.measured_pl_db))
    check(name+' stored squared error dB2',close(pred.squared_error_db2,(pred.predicted_pl_db-pred.measured_pl_db)**2))
    for key,g in pred.groupby(KEYS):
        proto=key[2];label=name+' '+str(key)
        check(label+' no duplicate held-out row',not g.row_id.duplicated().any())
        expected=5 if proto=='EXTERNAL_TEKBIYIK' else 34 if proto=='HOLDOUT_50_80' else 153
        check(label+' total row count',len(g)==expected)
        if proto=='EXTERNAL_TEKBIYIK':continue
        check(label+' coordinates/targets',all(int(r.row_id) in base.index and close([r.distance_m,r.angle_deg,r.measured_pl_db],base.loc[int(r.row_id),['distance_m','angle_deg','measured_pl_db']].to_numpy(dtype=float)) for r in g.itertuples()))
        for fold,h in g.groupby('fold_id'):
            test=set(h.row_id.astype(int));train=set(range(153))-test
            if proto=='LOO':expected_train=152;expected_test=1
            else:
                item=manifest[(proto,fold)];train=set(item['training_row_ids']);check(label+' '+fold+' exact test membership',test==set(item['test_row_ids']))
                expected_train=len(train);expected_test=len(test)
            check(label+' '+fold+' disjoint train/test',not (train&test) and len(train|test)==153)
            check(label+' '+fold+' counts',h.train_count.eq(expected_train).all() and h.test_count.eq(expected_test).all())
            if proto=='LODO':check(label+' '+fold+' distance group isolation',len(h.distance_m.unique())==1 and not base.loc[list(train)].distance_m.isin(h.distance_m).any())
            if proto=='LOAO':check(label+' '+fold+' angle group isolation',len(h.angle_deg.unique())==1 and not base.loc[list(train)].angle_deg.isin(h.angle_deg).any())
            if 'predictive_std_db' in h and h.predictive_std_db.notna().all():
                sd=h.predictive_std_db.to_numpy();lo=h.predicted_pl_db-1.645*sd;hi=h.predicted_pl_db+1.645*sd
                check(label+' '+fold+' finite nonnegative observation std',np.isfinite(sd).all() and (sd>=0).all())
                for col,calc in [('lower90_db',lo),('upper90_db',hi),('half_width_db',1.645*sd),('full_width_db',hi-lo)]:check(label+' '+fold+' '+col,close(h[col],calc))
                cov=(h.measured_pl_db>=lo)&(h.measured_pl_db<=hi)
                stored=h.covered.map({True:True,False:False,'True':True,'False':False})
                check(label+' '+fold+' coverage flags',np.array_equal(stored.to_numpy(),cov.to_numpy()))
                score=hi-lo+20*np.maximum(lo-h.measured_pl_db,0)+20*np.maximum(h.measured_pl_db-hi,0)
                check(label+' '+fold+' interval scores',close(h.interval_score_db,score))
    return pred

def main(full=False,preserve=False):
    """Verify all available raw numerical artifacts, optionally final synthesis/preservation."""
    base=read('00_baseline/pl_targets_153.csv').set_index('row_id')
    check('Original 9x17 nonuniform measurement grid',len(base)==153 and np.array_equal(np.sort(base.angle_deg.unique()),np.r_[np.arange(16),20]) and np.array_equal(np.sort(base.distance_m.unique()),np.arange(20,101,10)/100))
    manifest={(x['protocol'],x['fold_id']):x for x in json.loads((FINAL/'99_provenance/fold_manifest.json').read_text(encoding='utf-8'))}
    for key,item in manifest.items():check('manifest '+str(key),not(set(item['training_row_ids'])&set(item['test_row_ids'])) and set(item['training_row_ids'])|set(item['test_row_ids'])==set(range(153)))
    bundles=[('01_m5_grouped/m5_all_predictions.csv','01_m5_grouped/m5_fold_metrics.csv','01_m5_grouped/m5_grouped_summary.csv'),
        ('02_grouped_uncertainty/m4_grouped_pi_predictions.csv','02_grouped_uncertainty/grouped_pi_fold_metrics.csv','02_grouped_uncertainty/grouped_pi_summary.csv'),
        ('00_baseline/recovered_comparison_predictions.csv','00_baseline/recovered_fold_metrics_corrected.csv','00_baseline/recovered_summary_corrected.csv')]
    for rawpath,foldpath,sumpath in bundles:
        pred=verify_predictions(rawpath,manifest,base)
        if rawpath.startswith('02_'):
            m5=verify_predictions('02_grouped_uncertainty/m5_grouped_pi_predictions.csv',manifest,base)
            check('M5 uncertainty uses exact Experiment A predictions',close(m5.predicted_pl_db,read('01_m5_grouped/m5_all_predictions.csv').predicted_pl_db) and close(m5.predictive_std_db,read('01_m5_grouped/m5_all_predictions.csv').predictive_std_db))
            pred=pd.concat([pred,m5],ignore_index=True)
        ef,es=independent_tables(pred);compare(ef,read(foldpath),KEYS+['fold_id'],foldpath);compare(es,read(sumpath),KEYS+['aggregation'],sumpath)
    ext=verify_predictions('00_baseline/historical_external_predictions.csv',manifest,base)
    _,es=independent_tables(ext);compare(es,read('00_baseline/historical_external_summary.csv'),KEYS+['aggregation'],'historical external')
    for filename in ['m5_lodo_predictions.csv','m5_loao_predictions.csv','m5_50_80_predictions.csv']:
        p=read('01_m5_grouped/'+filename);source=read('01_m5_grouped/m5_all_predictions.csv').set_index('prediction_id').loc[p.prediction_id]
        check(filename+' exact subset',close(p.predicted_pl_db,source.predicted_pl_db))
    pi=read('02_grouped_uncertainty/grouped_pi_interval_scores.csv');gpraw=pd.concat([read('02_grouped_uncertainty/m4_grouped_pi_predictions.csv'),read('02_grouped_uncertainty/m5_grouped_pi_predictions.csv')]).set_index('prediction_id').loc[pi.prediction_id]
    check('interval-score table raw equivalence',close(pi.interval_score_db,gpraw.interval_score_db))
    repro=read('02_grouped_uncertainty/m4_historical_reproduction.csv')
    if (FINAL/'02_grouped_uncertainty/grouped_pi_reporting_metrics.csv').exists():
        reporting=read('02_grouped_uncertainty/grouped_pi_reporting_metrics.csv')
        check('Reported coverage percentages',close(reporting.coverage_percent,100*reporting.covered_count/reporting.coverage_denominator))
    check('M4 every frozen grouped fold within prespecified material tolerance',(repro.absolute_difference_db<=.001).all() and len(repro)==27)
    check('M4 reproduction differences independently recomputed',close(repro.absolute_difference_db,abs(repro.reproduced_rmse_db-repro.frozen_rmse_db)))
    for path in ['01_m5_grouped/m5_optimized_kernels.csv','02_grouped_uncertainty/m4_optimized_kernels.csv']:
        k=read(path);check(path+' exact grouped GP settings',k.alpha.eq(0).all() and k.n_restarts_optimizer.eq(2).all() and k.random_state.eq(42).all() and k.normalize_y.all())
        check(path+' observation noise in response units',close(k.observation_noise_variance_db2,k.noise_variance_normalized*k.response_train_std_db**2))
        check(path+' fold variance algebra',(k.variance_formula_max_abs_error_db2<1e-8).all())
    hp=read('01_m5_grouped/m5_50_80_predictions.csv');hs=read('01_m5_grouped/m5_holdout_distance_metrics.csv')
    for row in hs.itertuples():
        vals=calculated(hp[np.isclose(hp.distance_m,row.subset_distance_m)])
        for c,v in vals.items():check('M5 holdout subset '+str(row.subset_distance_m)+' '+c,close(getattr(row,c),v))
    if full:
        mlp=verify_predictions('03_mlp_robustness/mlp_fold_predictions.csv',manifest,base);ef,es=independent_tables(mlp)
        savedfold=read('03_mlp_robustness/mlp_fold_metrics.csv');compare(ef,savedfold,KEYS+['fold_id'],'MLP folds');compare(es,read('03_mlp_robustness/mlp_configuration_summary.csv'),KEYS+['aggregation'],'MLP summary')
        cfg=read('03_mlp_robustness/mlp_configurations.csv');check('Full prespecified MLP grid',len(cfg)==30 and len(savedfold)==810 and len(mlp)==30*(153+153+34))
        check('MLP failures retained and no fits silently discarded',savedfold.status.eq('completed').all())
        check('MLP original baseline uniquely identified',int(cfg.published_baseline.sum())==1)
        seed=read('03_mlp_robustness/mlp_seed_summary.csv');sm=read('03_mlp_robustness/mlp_configuration_summary.csv');pool=sm[sm.aggregation=='pooled_points']
        for r in seed.itertuples():
            g=pool[(pool.architecture==r.architecture)&(pool.alpha==r.alpha)&(pool.protocol==r.protocol)]
            for c,v in dict(n_seeds=len(g),mean_seed_pooled_rmse_db=g.rmse_db.mean(),sd_seed_pooled_rmse_db=g.rmse_db.std(ddof=1),min_seed_pooled_rmse_db=g.rmse_db.min(),max_seed_pooled_rmse_db=g.rmse_db.max(),min_seed_mae_db=g.mae_db.min(),max_seed_mae_db=g.mae_db.max()).items():check('MLP seed '+str((r.architecture,r.alpha,r.protocol))+' '+c,close(getattr(r,c),v))
        factor=read('04_scientific_synthesis/mlp_factor_metrics.csv')
        for r in factor.itertuples():
            g=pool[(pool.architecture==r.architecture)&(pool.alpha==r.alpha)&(pool.protocol==r.protocol)]
            for c,v in dict(n_seeds=len(g),mean_pooled_rmse_db=g.rmse_db.mean(),min_pooled_rmse_db=g.rmse_db.min(),max_pooled_rmse_db=g.rmse_db.max(),sd_seed_rmse_db=g.rmse_db.std(ddof=1)).items():check('MLP factors '+str((r.architecture,r.alpha,r.protocol))+' '+c,close(getattr(r,c),v))
        pairs=read('04_scientific_synthesis/mlp_regularization_pairs.csv')
        for r in pairs.itertuples():
            g=pool[(pool.architecture==r.architecture)&(pool.seed==r.seed)&(pool.protocol==r.protocol)]
            lo=g[g.alpha==.001].rmse_db.iloc[0];hi=g[g.alpha==.01].rmse_db.iloc[0]
            check('MLP matched alpha pair '+str((r.architecture,r.seed,r.protocol)),close(r.rmse_high_minus_low_db,hi-lo))
        for r in read('04_scientific_synthesis/mlp_effect_metrics.csv').itertuples():
            g=pool[pool.protocol==r.protocol];a=g.groupby('architecture').rmse_db.mean()
            ranges=[h.rmse_db.max()-h.rmse_db.min() for _,h in g.groupby(['architecture','alpha'])]
            changes=pairs[pairs.protocol==r.protocol].rmse_high_minus_low_db
            vals=dict(architecture_mean_pooled_rmse_db=a.loc[r.architecture],minimum_architecture_mean_rmse_db=a.min(),maximum_architecture_mean_rmse_db=a.max(),maximum_within_architecture_alpha_seed_range_db=max(ranges),median_absolute_paired_alpha_change_db=np.median(abs(changes)),maximum_absolute_paired_alpha_change_db=max(abs(changes)),mean_signed_paired_alpha_change_db=changes.mean())
            for c,v in vals.items():check('MLP effects '+str((r.architecture,r.protocol))+' '+c,close(getattr(r,c),v))
        descriptive=read('04_scientific_synthesis/mlp_descriptive_comparison.csv');reference=read('00_baseline/recovered_summary_corrected.csv')
        for r in descriptive.itertuples():
            g=pool[pool.protocol==r.protocol];b=g[g.published_baseline].rmse_db.iloc[0]
            m1=reference[(reference.model_name=='M1')&(reference.protocol==r.protocol)&(reference.aggregation=='pooled_points')].rmse_db.iloc[0]
            m2=reference[(reference.model_name=='M2')&(reference.protocol==r.protocol)&(reference.aggregation=='pooled_points')].rmse_db.iloc[0]
            vals=dict(n_configurations=len(g),minimum_descriptive_pooled_rmse_db=g.rmse_db.min(),maximum_descriptive_pooled_rmse_db=g.rmse_db.max(),median_configuration_pooled_rmse_db=g.rmse_db.median(),baseline_pooled_rmse_db=b,baseline_ascending_rank=1+(g.rmse_db<b).sum(),configurations_below_m1=(g.rmse_db<m1).sum(),configurations_below_m2=(g.rmse_db<m2).sum(),m1_pooled_rmse_db=m1,m2_pooled_rmse_db=m2)
            for c,v in vals.items():check('MLP descriptive '+r.protocol+' '+c,close(getattr(r,c),v))
        runtime=read('04_scientific_synthesis/runtime_metrics.csv').iloc[0]
        for c,v in dict(sweep_fits=len(savedfold),successful_fits=savedfold.status.eq('completed').sum(),failed_fits=savedfold.status.eq('failed').sum(),max_iter_fits=savedfold.hit_max_iter.sum(),total_fit_seconds=savedfold.fit_seconds.sum(),median_fit_seconds=savedfold.fit_seconds.median(),maximum_fit_seconds=savedfold.fit_seconds.max()).items():check('runtime '+c,close(runtime[c],v))
        m3=read('03_mlp_robustness/m3_historical_reproduction.csv');check('Original M3 reproduction equivalent',(m3.absolute_difference_db<=.001).all() and len(m3)==27)
        originals=mlp[mlp.configuration_id==cfg.loc[cfg.published_baseline,'configuration_id'].iloc[0]].copy();originals['model_name']='M3'
        polynomials=read('00_baseline/recovered_comparison_predictions.csv');polynomials=polynomials[polynomials.protocol.isin(['LODO','LOAO'])]
        support_raw=pd.concat([polynomials,read('02_grouped_uncertainty/m4_grouped_pi_predictions.csv'),read('02_grouped_uncertainty/m5_grouped_pi_predictions.csv'),originals],ignore_index=True)
        support_raw['support_geometry']=[manifest[(p,f)]['support_geometry'] for p,f in zip(support_raw.protocol,support_raw.fold_id)]
        for r in read('04_scientific_synthesis/support_geometry_metrics.csv').itertuples():
            g=support_raw[(support_raw.model_name==r.model_name)&(support_raw.protocol==r.protocol)&(support_raw.support_geometry==r.support_geometry)]
            per=[calculated(h)['rmse_db'] for _,h in g.groupby('fold_id')];vals=calculated(g)
            check('support geometry '+str((r.model_name,r.protocol,r.support_geometry))+' pooled error',close(r.pooled_rmse_db,vals['rmse_db']))
            check('support geometry '+str((r.model_name,r.protocol,r.support_geometry))+' macro error',close(r.macro_rmse_db,np.mean(per)))
            if r.model_name in ['M4','M5']:
                for c in ['covered_count','coverage_denominator','coverage','mean_full_width_db','interval_score_db']:check('support '+str((r.model_name,r.protocol,r.support_geometry))+' '+c,close(getattr(r,c),vals[c]))
        penalties=read('04_scientific_synthesis/separability_raw.csv');matrix=base.measured_pl_db.to_numpy().reshape(9,17);delta=matrix-matrix[:,0:1]
        check('separability raw penalties recomputed',close(penalties.angular_penalty_db,delta.ravel()))
        check('separability centered penalties recomputed',close(penalties.centered_angular_penalty_db,(delta-delta.mean(axis=0)).ravel()))
        for j,r in enumerate(read('04_scientific_synthesis/separability_metrics.csv').itertuples()):
            check('separability range '+str(r.angle_deg),close(r.cross_distance_range_db,max(delta[:,j])-min(delta[:,j])))
            check('separability centered maximum '+str(r.angle_deg),close(r.maximum_absolute_centered_db,max(abs(delta[:,j]-delta[:,j].mean()))))
        unified=read('04_scientific_synthesis/UNIFIED_MODEL_COMPARISON.csv')
        for r in unified.itertuples():
            if r.evidence_status=='unavailable':continue
            src=read(r.source_file);origin=src.iloc[int(r.source_row_zero_based)]
            for c in ['rmse_db','mae_db','coverage','mean_full_width_db','interval_score_db']:
                if c in src and c in unified:check('unified '+str((r.model_name,r.configuration_id,r.protocol,r.aggregation))+' '+c,close(getattr(r,c),origin[c]))
        ledger=read('04_scientific_synthesis/NUMERICAL_CLAIM_LEDGER.csv');cache={}
        for r in ledger.itertuples():
            if r.source_file not in cache:cache[r.source_file]=read(r.source_file)
            source=cache[r.source_file];row=int(json.loads(r.source_column_or_rows)['row_zero_based']);col=json.loads(r.source_column_or_rows)['column']
            check('claim '+r.claim_id,close(r.value_full_precision,source.iloc[row][col]))
    if preserve:
        sources=read('00_baseline/source_manifest.csv');changes=[]
        for r in sources.itertuples():
            h=hashlib.sha256()
            with Path(r.source_path).open('rb') as stream:
                for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
            if h.hexdigest()!=r.sha256:changes.append(r.source_path)
        check('All original project files preserved by SHA256',not changes,json.dumps(changes))
    report='# Independent numerical verification\n\nNo model fitting objects or fitting-side metric functions are used. The calculations read saved observation/prediction rows. Tolerance: rtol=1e-9, atol=1e-10; material historical GP tolerance=0.001 dB.\n\n'
    report+='| Test | Status | Detail |\n|---|---|---|\n'+'\n'.join(f"| {t['test']} | {t['status']} | {t['detail']} |" for t in TESTS)
    failed=[t for t in TESTS if t['status']=='FAIL'];report+=f'\n\nTests: {len(TESTS)}; failures: {len(failed)}.\n'
    destination=FINAL/'99_provenance'/('numerical_verification.md' if full else 'numerical_verification_core.md')
    with destination.open('x',encoding='utf-8') as stream:stream.write(report)
    with destination.with_suffix('.json').open('x',encoding='utf-8') as stream:json.dump(dict(tests=TESTS,failed_count=len(failed),full=full,preservation_checked=preserve),stream,indent=2)
    print('Independent verification:',len(TESTS),'tests;',len(failed),'failures',flush=True)
    for failure in failed:print(failure,flush=True)
    if failed:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');parser.add_argument('--preserve',action='store_true');args=parser.parse_args();main(args.full,args.preserve)
