"""Prespecified 30-configuration descriptive sensitivity study; no model selection."""
from pathlib import Path
import sys
import time
import warnings
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'99_provenance'))
from evidence_common import *
PRACTICAL_BUDGET_SECONDS=3600  # pilot-based conservative estimate threshold, declared before sweep

def configurations():
    """Exact authorized grid; retain every sklearn constructor/default parameter."""
    f=original_functions();rows=[]
    for arch in [(16,8),(32,16),(64,32)]:
        for alpha in [.001,.01]:
            for seed in [0,1,2,3,42]:
                config=f'MLP_{arch[0]}_{arch[1]}_a{alpha}_s{seed}'
                pipe=f['build_mlp'](hidden=arch,alpha=alpha);pipe.set_params(mlp__random_state=seed)
                rows.append(dict(configuration_id=config,config_id=config,architecture=str(arch),hidden1=arch[0],hidden2=arch[1],alpha=alpha,seed=seed,
                    parameter_count=3*arch[0]+(arch[0]+1)*arch[1]+arch[1]+1,
                    published_baseline=(arch==(32,16) and alpha==.01 and seed==42),
                    pipeline_parameters_json=json.dumps({k:str(v) for k,v in pipe.get_params(deep=True).items()}),
                    response_transform='none',feature_transform='log10(d/0.20),theta/20; training-fold StandardScaler',
                    sensitivity_only=True,unbiased_selected_test_performance=False))
    return pd.DataFrame(rows)

def one_fit(c,df,X,y,foldtuple):
    """Fit one frozen pipeline; retain termination and all warnings, including failures."""
    protocol,fold,train,test=foldtuple;start=time.perf_counter();f=original_functions()
    pipe=f['build_mlp'](hidden=(int(c.hidden1),int(c.hidden2)),alpha=float(c.alpha));pipe.set_params(mlp__random_state=int(c.seed))
    meta=dict(configuration_id=c.configuration_id,config_id=c.configuration_id,architecture=c.architecture,alpha=float(c.alpha),seed=int(c.seed),
        protocol=protocol,fold_id=fold,model_name='MLP_sensitivity',train_count=len(train),test_count=len(test))
    pred=None;warn=[]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');pipe.fit(X[train],y[train]);mu=pipe.predict(X[test])
        warn=warning_rows(caught,'MLP_sensitivity',protocol,fold,c.configuration_id)
        mlp=pipe.named_steps['mlp'];assert np.isfinite(mu).all()
        pred=prediction_rows(df,mu,protocol,fold,train,test,'MLP_sensitivity',c.configuration_id,
            extra=dict(config_id=c.configuration_id,architecture=c.architecture,alpha=float(c.alpha),seed=int(c.seed)))
        meta.update(metric_values(pred),status='completed',n_iter=int(mlp.n_iter_),training_loss=float(mlp.loss_),
            best_validation_score=float(mlp.best_validation_score_),hit_max_iter=bool(mlp.n_iter_>=mlp.max_iter),
            termination='max_iter' if mlp.n_iter_>=mlp.max_iter else 'early_stopping',
            warning_count=len(warn),failure_reason='',scaler_mean_json=json.dumps(pipe.named_steps['scaler'].mean_.tolist()),
            scaler_scale_json=json.dumps(pipe.named_steps['scaler'].scale_.tolist()))
    except Exception as e:
        meta.update(status='failed',failure_reason=repr(e))
    meta['fit_seconds']=time.perf_counter()-start
    return pred,meta,warn

def pilot():
    """Benchmark every architecture on sparse-angle/boundary-distance folds, baseline seed."""
    out=FINAL/'03_mlp_robustness';df,X,y=targets();cfg=configurations();write_csv(cfg,out/'mlp_configurations.csv')
    selected=[t for t in folds(df) if (t[0],t[1]) in [('LODO','20cm'),('LOAO','20deg')]]
    rows=[];pred=[];warn=[]
    for _,c in cfg[(cfg.alpha==.01)&(cfg.seed==42)].iterrows():
        for fold in selected:
            p,m,w=one_fit(c,df,X,y,fold);rows.append(m);warn.extend(w)
            if p is not None:pred.append(p)
    write_csv(pd.DataFrame(rows),out/'mlp_pilot_fit_times.csv');write_csv(pd.concat(pred,ignore_index=True),out/'mlp_pilot_predictions.csv');write_json(warn,out/'mlp_pilot_warnings.json')
    maxfit=max(r['fit_seconds'] for r in rows);estimate=810*maxfit*1.5
    gate=dict(pilot_count=len(rows),fit_count=810,max_pilot_fit_seconds=maxfit,conservative_estimate_seconds=estimate,
        practical_budget_seconds=PRACTICAL_BUDGET_SECONDS,pass_budget=estimate<=PRACTICAL_BUDGET_SECONDS,
        all_pilot_fits_completed=all(r['status']=='completed' for r in rows),
        methodology='six fits; all architectures, alpha=.01, seed=42; LODO20cm and LOAO20deg; maximum observed fit time times 810 times 1.5 safety factor')
    write_json(gate,out/'mlp_budget_gate.json');print(json.dumps(gate,indent=2),flush=True)

def sweep():
    """Run all 810 authorized fits only after a passed practical-cost gate."""
    out=FINAL/'03_mlp_robustness';gate=json.loads((out/'mlp_budget_gate.json').read_text(encoding='utf-8'))
    if not gate['pass_budget'] or not gate['all_pilot_fits_completed']:raise RuntimeError('MLP budget/pipeline gate blocked; user approval required for a reduced alternative.')
    cfg=pd.read_csv(out/'mlp_configurations.csv');df,X,y=targets();pred=[];rows=[];warn=[];start=time.perf_counter()
    for _,c in cfg.iterrows():
        for fold in folds(df):
            p,m,w=one_fit(c,df,X,y,fold);rows.append(m);warn.extend(w)
            if p is not None:pred.append(p)
        print(c.configuration_id,len(rows),'fits',time.perf_counter()-start,'s',flush=True)
    write_csv(pd.DataFrame(rows),out/'mlp_fold_metrics.csv');write_json(warn,out/'mlp_warnings.json')
    allpred=pd.concat(pred,ignore_index=True);write_csv(allpred,out/'mlp_fold_predictions.csv')
    fm,sm=tables(allpred)
    extras=[]
    for (config,protocol),g in pd.DataFrame(rows).groupby(['configuration_id','protocol']):
        extras.append(dict(configuration_id=config,protocol=protocol,fit_seconds=float(g.fit_seconds.sum()),
            completed_fits=int((g.status=='completed').sum()),failed_fits=int((g.status=='failed').sum()),
            max_iter_fits=int(g.hit_max_iter.fillna(False).sum())))
    sm=sm.merge(pd.DataFrame(extras),on=['configuration_id','protocol']).merge(cfg[['configuration_id','architecture','alpha','seed','published_baseline']],on='configuration_id')
    write_csv(sm,out/'mlp_configuration_summary.csv')
    seed=[]
    pooled=sm[sm.aggregation=='pooled_points']
    for (architecture,alpha,protocol),g in pooled.groupby(['architecture','alpha','protocol']):
        seed.append(dict(architecture=architecture,alpha=alpha,protocol=protocol,n_seeds=len(g),
            mean_seed_pooled_rmse_db=float(g.rmse_db.mean()),sd_seed_pooled_rmse_db=float(g.rmse_db.std(ddof=1)),
            min_seed_pooled_rmse_db=float(g.rmse_db.min()),max_seed_pooled_rmse_db=float(g.rmse_db.max()),
            min_seed_mae_db=float(g.mae_db.min()),max_seed_mae_db=float(g.mae_db.max())))
    write_csv(pd.DataFrame(seed),out/'mlp_seed_summary.csv');print('SWEEP COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','sweep']);args=parser.parse_args()
    pilot() if args.stage=='pilot' else sweep()
