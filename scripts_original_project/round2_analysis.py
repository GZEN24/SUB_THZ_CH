"""Round-2 evidence: subbands, saved CV audit, and floating-intercept LoS refits.

Python 3.10+, NumPy/SciPy/pandas/Matplotlib. Original scripts/data are read only.
Outputs are exclusive-create; a second run cannot replace existing evidence.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'block2_output' / 'round2_revision'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mplconfig'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import linregress, t
import matplotlib.pyplot as plt
import block2_complete as core


def csv_new(frame: pd.DataFrame, name: str) -> None:
    """Write a new UTF-8 evidence table without replacing an artifact."""
    with (OUT / name).open('x', encoding='utf-8', newline='') as stream:
        frame.to_csv(stream, index=False)


def json_new(value: dict, name: str) -> None:
    """Write a new JSON evidence artifact."""
    with (OUT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def fit_los(d: np.ndarray, y: np.ndarray, label: str) -> dict:
    """OLS PL=a log10(d/m)+b; Student-t interval assumes iid Gaussian errors."""
    x = np.log10(d)
    w = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    pred = w[0] * x + w[1]
    reg = linregress(x, y)
    se = reg.stderr / 10
    half = t.ppf(0.975, len(d) - 2) * se
    return dict(fit=label, n_points=len(d), slope_db_per_decade=w[0],
                n=w[0]/10, intercept_at_1m_db=w[1],
                fitted_reference_at_20cm_db=w[0]*np.log10(.2)+w[1],
                rmse_db=float(np.sqrt(np.mean((y-pred)**2))), r2=reg.rvalue**2,
                n_standard_error=se, n_ci95_low=w[0]/10-half,
                n_ci95_high=w[0]/10+half,
                ci_assumption='conditional OLS iid homoscedastic errors; no calibration uncertainty')


def main() -> None:
    """Generate R2.1/R2.2/R2.4 evidence using measured CFR and frozen CV results."""
    OUT.mkdir(exist_ok=True, parents=True)
    data = core.load_all_karakoca(ROOT/'Datasets/Sub_THz_measurements')
    d, a, pl, ds, bc = core.extract_statistics(data, angle_cap=core.ANGLE_CAP_DEG)
    rows, band_meta, interaction, fits, complex_checks = [], [], [], [], []
    all_band_pl = []
    for item in data:
        f = item['freqs_hz']
        assert np.all(np.diff(f)>0) and np.isclose(f[0],240e9) and np.isclose(f[-1],300e9)
        m = sio.loadmat(ROOT/'Datasets/Sub_THz_measurements'/f"{int(round(item['distance_m']*100))}cm.mat",
                       variable_names=['channel_frequency_response','file_name_list'])
        order = np.argsort(core._col_to_angle_map(m))
        db_complex = 20*np.log10(np.abs(m['channel_frequency_response'][:,order]))
        error = float(np.max(np.abs(db_complex-item['cfr_db'])))
        complex_checks.append(error)
        assert np.allclose(db_complex,item['cfr_db'],atol=1e-10,rtol=1e-10)
        sub = []
        for k,(lo,hi) in enumerate([(240,255),(255,270),(270,285),(285,300)]):
            mask=(f>=lo*1e9)&((f<hi*1e9) if k<3 else (f<=hi*1e9))
            assert mask.sum()==1024
            sub.append(-np.mean(item['cfr_db'][mask],axis=0))
            if item is data[0]:band_meta.append(dict(subband=k+1,low_ghz=lo,high_ghz=hi,
                n_frequencies=int(mask.sum()),first_hz=float(f[mask][0]),last_hz=float(f[mask][-1])))
        sub=np.array(sub).T
        full=-np.mean(item['cfr_db'],axis=0)
        assert np.allclose(full,np.mean(sub,axis=1),atol=1e-10)
        all_band_pl.append(sub)
        for j,theta in enumerate(item['angles_deg']):
            row=dict(distance_m=item['distance_m'],angle_deg=int(theta),retained_for_model=bool(theta<=20),
                     full_band_pl_db=full[j],subband_range_db=float(np.ptp(sub[j])),
                     angular_penalty_subband_range_db=float(np.ptp(sub[j]-sub[0])),
                     full_band_angular_penalty_db=full[j]-full[0])
            for k in range(4):
                row.update({f'band{k+1}_pl_db':sub[j,k],f'band{k+1}_minus_full_db':sub[j,k]-full[j],
                            f'band{k+1}_angular_penalty_db':sub[j,k]-sub[0,k]})
            rows.append(row)
    frame=pd.DataFrame(rows)
    csv_new(frame,'r2_1_subband_sensitivity.csv')
    cube=np.array(all_band_pl)[:,data[0]['angles_deg']<=20,:]
    for k in range(4):
        penalty=cube[:,:,k]-cube[:,0:1,k]
        deviation=penalty-penalty.mean(axis=0)
        peak=np.max(np.abs(deviation),axis=0)
        for j,theta in enumerate(a):interaction.append(dict(subband=k+1,angle_deg=int(theta),
            max_abs_centered_penalty_db=peak[j],mean_penalty_db=penalty[:,j].mean()))
        fits.append(fit_los(d,cube[:,0,k],f'subband_{k+1}'))
    csv_new(pd.DataFrame(interaction),'r2_1_subband_interaction.csv')
    csv_new(pd.DataFrame(fits),'r2_1_subband_los_fits.csv')
    summaries={}
    for name,s in [('all_171',frame),('retained_153',frame[frame.retained_for_model]),('theta20',frame[frame.angle_deg==20])]:
        delta=s[[f'band{k}_minus_full_db' for k in range(1,5)]].to_numpy()
        summaries[name]=dict(n_points=len(s),median_range_db=float(s.subband_range_db.median()),
            maximum_range_db=float(s.subband_range_db.max()),max_abs_subband_minus_full_db=float(np.max(np.abs(delta))),
            rms_subband_minus_full_db=float(np.sqrt(np.mean(delta**2))),
            median_angular_penalty_range_db=float(s.angular_penalty_subband_range_db.median()),
            maximum_angular_penalty_range_db=float(s.angular_penalty_subband_range_db.max()))
    summaries['subbands']=band_meta
    summaries['max_complex_vs_stored_db_error']=max(complex_checks)
    summaries['frequency_spacing_mhz']=float(np.diff(data[0]['freqs_hz'])[0]/1e6)
    summaries['peak_interaction_angle_by_subband']=[int(a[np.argmax(np.array([r['max_abs_centered_penalty_db'] for r in interaction if r['subband']==k]))]) for k in range(1,5)]
    json_new(summaries,'r2_1_metrics.json')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.1),layout='constrained')
    for k in range(4):
        values=np.array([r['max_abs_centered_penalty_db'] for r in interaction if r['subband']==k+1])
        axes[0].plot(a,values,'o-',ms=3,label=f'{240+15*k}–{255+15*k} GHz')
        axes[1].plot(d*100,cube[:,-1,k]-pl[:,-1],'o-',ms=3)
    axes[0].set(xlabel='Angle (deg)',ylabel='Max centered angular penalty (dB)')
    axes[0].axvspan(8,13,alpha=.1,color='gray'); axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Distance (cm)',ylabel='Subband − full-band PL (dB)',title='20° misalignment')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'r2_1_subband_sensitivity.png',dpi=600);plt.close(fig)
    csv_new(pd.DataFrame({'distance_cm':d*100,'distance_m':d,'theta_deg':0,'measured_pl_db':pl[:,0]}),'r2_4_los_values.csv')
    f9=fit_los(d,pl[:,0],'all_nine'); f8=fit_los(d[1:],pl[1:,0],'exclude_20cm')
    np.testing.assert_allclose([f9['slope_db_per_decade'],f9['intercept_at_1m_db']],np.load(ROOT/'block2_output/sep_w_los.npy'),atol=1e-10)
    csv_new(pd.DataFrame([f9,f8]),'r2_4_nearfield_refit.csv')
    fig,ax=plt.subplots(figsize=(3.5,2.8),layout='constrained');ax.plot(d*100,pl[:,0],'ko',ms=4,label='Measured')
    for fit in [f9,f8]:ax.plot(d*100,fit['slope_db_per_decade']*np.log10(d)+fit['intercept_at_1m_db'],label=f"{fit['n_points']} points: n={fit['n']:.4f}")
    ax.set(xlabel='Distance (cm)',ylabel='LoS path loss (dB)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUT/'r2_4_nearfield_refit.png',dpi=600);plt.close(fig)
    summary=pd.read_csv(ROOT/'block2_output/lodo_loao_summary.csv').set_index('Protocol')
    lodo=pd.read_csv(ROOT/'block2_output/lodo_per_fold.csv');loao=pd.read_csv(ROOT/'block2_output/loao_per_fold.csv')
    sensitivity=pd.read_csv(ROOT/'block2_output/gpr_kernel_sensitivity_results.csv')
    loo={'M1':.411,'M2':.224,'M3':.561,'M4':float(sensitivity.iloc[0].LOO_RMSE_dB)}
    audit=[]
    for model in loo:
        np.testing.assert_allclose([lodo[model].mean(),loao[model].mean()],summary[model].values,atol=1e-12)
        row=dict(model=model,loo_rmse_db=loo[model],lodo_avg_rmse_db=summary.loc['LODO avg',model],
                 loao_avg_rmse_db=summary.loc['LOAO avg',model],
                 loo_source='frozen source/manuscript rounded reference' if model!='M4' else 'frozen kernel sensitivity CSV')
        row['lodo_over_loo']=row['lodo_avg_rmse_db']/loo[model];row['loao_over_loo']=row['loao_avg_rmse_db']/loo[model]
        for _,fold in lodo.iterrows():row[f'lodo_{int(fold.distance_cm)}cm_rmse_db']=fold[model]
        audit.append(row)
    csv_new(pd.DataFrame(audit),'r2_2_cv_spatial_support.csv')
    csv_new(pd.DataFrame([{'distance_m':dd,'angle_deg':aa,'pl_db':pl[i,j],'ds_ns':ds[i,j],'bc_mhz':bc[i,j]}
                         for i,dd in enumerate(d) for j,aa in enumerate(a)]),'channel_statistics.csv')
    print(json.dumps({'subband':summaries,'los_refits':[f9,f8]},indent=2))


if __name__=='__main__':
    main()
