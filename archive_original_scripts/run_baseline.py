"""Recover frozen evidence and extract PL targets without training models."""
from pathlib import Path
import sys
import ast
import inspect
import json
import platform
import os
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'99_provenance'))
from evidence_common import *
import scipy
import sklearn
from threadpoolctl import threadpool_info
from pypdf import PdfReader

def main():
    """Hash preserved inputs, inspect PDF, reconstruct the original observation grid."""
    out=FINAL/'00_baseline'
    for name in ['00_baseline','01_m5_grouped','02_grouped_uncertainty','03_mlp_robustness','04_scientific_synthesis','99_provenance']:(FINAL/name).mkdir(exist_ok=True)
    original=[]
    for path in ROOT.rglob('*'):
        if not path.is_file():continue
        if FINAL in path.parents and FINAL/'Last_paper' not in path.parents:continue
        stat=path.stat();original.append(dict(source_path=str(path),relative_path=str(path.relative_to(ROOT)),size_bytes=stat.st_size,
            modified_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),sha256=sha256(path)))
    manifest=pd.DataFrame(original);write_csv(manifest,out/'source_manifest.csv')
    write_text('\n'.join(f"{r.sha256}  {r.source_path}" for r in manifest.itertuples()),FINAL/'99_provenance/input_hashes.sha256')
    env=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        pandas=pd.__version__,platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),threadpools=threadpool_info())
    write_text(json.dumps(env,indent=2),out/'environment.txt')
    f=original_functions();data=f['load_all_karakoca'](ROOT/'Datasets/Sub_THz_measurements')
    d=np.array([x['distance_m'] for x in data]);a=np.r_[np.arange(16),20]
    assert np.array_equal(d,np.arange(20,101,10)/100)
    rows=[]
    for i,item in enumerate(data):
        for j,angle in enumerate(a):
            col=int(np.flatnonzero(item['angles_deg']==angle)[0]);path=ROOT/'Datasets/Sub_THz_measurements'/f'{int(round(d[i]*100))}cm.mat'
            rows.append(dict(row_id=i*len(a)+j,distance_m=d[i],angle_deg=angle,measured_pl_db=float(-np.mean(item['cfr_db'][:,col])),
                source_file=str(path),source_sha256=manifest.loc[manifest.source_path==str(path),'sha256'].iloc[0],source_sorted_column=col,
                target_definition='negative arithmetic mean of 4096 CFR dB values'))
    df=pd.DataFrame(rows);assert len(df)==153 and np.isfinite(df.measured_pl_db).all()
    saved=pd.read_csv(ROOT/'block2_output/round2_revision/channel_statistics.csv',float_precision='round_trip')
    np.testing.assert_allclose(df.measured_pl_db,saved.pl_db,atol=1e-12,rtol=0)
    write_csv(df,out/'pl_targets_153.csv')
    extracted=[]
    for filename in ['block2_complete.py','lodo_loao_cv.py','gpr_m5_matern_calibration.py','gpr_kernel_sensitivity.py']:
        source=(ROOT/filename).read_text(encoding='utf-8-sig');tree=ast.parse(source)
        extracted.extend([f'\n# {filename}\n'+ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef)])
    write_text('\n\n'.join(extracted),out/'audited_original_function_definitions.txt')
    write_text(inspect.getsource(GaussianProcessRegressor.predict)+'\n\n'+inspect.getsource(WhiteKernel.__call__)+'\n\n'+inspect.getsource(WhiteKernel.diag),out/'sklearn_predict_variance_source.txt')
    pdf=FINAL/'Last_paper/Ojcom.pdf';reader=PdfReader(pdf)
    write_text('\n\n'.join(f'PAGE {i+1}\n'+p.extract_text() for i,p in enumerate(reader.pages)),out/'authoritative_pdf_extracted.txt')
    reconcile=[]
    pred=pd.read_csv(ROOT/'block2_output/round2_revision/figure_loo_predictions.csv',float_precision='round_trip')
    np.testing.assert_allclose(pred.measured_pl_db,df.measured_pl_db,atol=1e-12,rtol=0)
    for model in ['M1','M2','M3','M4']:
        err=pred[f'{model}_prediction_db']-pred.measured_pl_db
        reconcile.append(dict(model=model,protocol='LOO',aggregation='pooled_points',metric='rmse_db',value=float(np.sqrt(np.mean(err**2))),source='block2_output/round2_revision/figure_loo_predictions.csv',evidence='raw rows available'))
    m5=pd.read_csv(ROOT/'block2_output/m5_matern_loo_predictions.csv',float_precision='round_trip');np.testing.assert_allclose(m5.y_true,df.measured_pl_db,atol=1e-12,rtol=0)
    reconcile.append(dict(model='M5',protocol='LOO',aggregation='pooled_points',metric='rmse_db',value=float(np.sqrt(np.mean((m5.y_pred_m5-m5.y_true)**2))),source='block2_output/m5_matern_loo_predictions.csv',evidence='raw rows available'))
    for protocol,file in [('LODO','lodo_per_fold.csv'),('LOAO','loao_per_fold.csv')]:
        p=pd.read_csv(ROOT/'block2_output'/file,float_precision='round_trip')
        for model in ['M1','M2','M3','M4']:
            reconcile.extend([dict(model=model,protocol=protocol,aggregation='macro_folds',metric='rmse_db',value=float(p[model].mean()),source=f'block2_output/{file}',evidence='fold RMSEs only; historical raw rows absent'),
                dict(model=model,protocol=protocol,aggregation='pooled_points',metric='rmse_db',value=float(np.sqrt(np.mean(p[model]**2))),source=f'block2_output/{file}',evidence='derived from equally sized historical folds; historical raw rows absent')])
    for model in ['M4','M5']:
        mu=pred.M4_prediction_db if model=='M4' else m5.y_pred_m5
        sd=pred.M4_std_db if model=='M4' else m5.y_std_m5
        yt=df.measured_pl_db
        for convention,h in [('fold_specific',Z90*sd),('constant_mean_halfwidth',float((Z90*sd).mean()))]:
            covered=int((np.abs(mu-yt)<=h).sum())
            reconcile.append(dict(model=model,protocol='LOO',aggregation=convention,metric='coverage',value=covered/len(yt),source='block2_output/round2_revision/figure_loo_predictions.csv' if model=='M4' else 'block2_output/m5_matern_loo_predictions.csv',evidence=f'{covered}/{len(yt)}'))
    write_csv(pd.DataFrame(reconcile),out/'baseline_reconciliation.csv')
    foldmanifest=[]
    for protocol,fold,train,test in folds(df):
        col='distance_m' if protocol!='LOAO' else 'angle_deg';support=sorted(df.iloc[train][col].unique().tolist())
        held=sorted(df.iloc[test][col].unique().tolist())
        geometry='boundary_extrapolation' if any(v<min(support) or v>max(support) for v in held) else 'interior_gap'
        foldmanifest.append(dict(protocol=protocol,fold_id=fold,training_row_ids=train.tolist(),test_row_ids=test.tolist(),
            held_out_values=held,support_axis=col,training_support_values=support,support_geometry=geometry,
            transformed_nearest_support_gap=[float(min(abs((np.log10(v/.2) if col=='distance_m' else v/20)-(np.log10(w/.2) if col=='distance_m' else w/20)) for w in support)) for v in held]))
    write_json(foldmanifest,FINAL/'99_provenance/fold_manifest.json')
    configs={m:{k:str(v) for k,v in gp(m).get_params(deep=True).items()} for m in ['M4','M5']}
    configs['M3_baseline']={k:str(v) for k,v in f['build_mlp']().get_params(deep=True).items()}
    write_json(configs,out/'frozen_model_configurations.json')
    print('GATE 0 PASS: targets agree with archived rows; baseline protocols established.',flush=True)
    print(pd.DataFrame(reconcile).to_string(index=False),flush=True)

if __name__=='__main__':main()
