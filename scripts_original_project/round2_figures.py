"""Regenerate the 13 manuscript plots from measured/fitted values in a new directory.

No original plot generator is executed. Numerical functions are reused from core.
PDF is vector; PNG is rendered at 600 dpi at intended one/two-column dimensions.
"""
from pathlib import Path
import os
import json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'block2_output/round2_revision'
FIG=OUT/'figures'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mplconfig'))
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text
from sklearn.gaussian_process import GaussianProcessRegressor
import block2_complete as core
import lodo_loao_cv as grouped
from round2_analysis import csv_new,json_new

COLORS=['#D55E00','#7755AA','#009E73','#CC3311']
NAMES=['M1','M2','M3','M4']
AUDIT=[]


def newfig(rows:int=1,cols:int=1,height:float=2.9):
    """Create an intended final-size one-column or two-column plot."""
    return plt.subplots(rows,cols,figsize=(7.16 if cols>1 else 3.5,height),layout='constrained',squeeze=False)


def save(fig,number:int,source:str) -> None:
    """Save vector PDF and genuine 600-dpi PNG, checking font sizes at final size."""
    for ax in fig.axes:
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10)
        ax.title.set_size(10)
        if ax.get_legend():
            for txt in ax.get_legend().get_texts():txt.set_fontsize(8.5)
        ax.grid(alpha=.18)
    fig.canvas.draw()
    for ext in ['pdf','png']:
        p=FIG/f'fig{number}.{ext}'
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,dpi=600)
    fonts=[t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    AUDIT.append(dict(figure=number,source=source,width_in=float(fig.get_figwidth()),height_in=float(fig.get_figheight()),
        png_dpi=600,min_text_pt=min(fonts),layout=f'{len(fig.axes)} axes including colorbars'))
    plt.close(fig)


def main() -> None:
    """Refit for fresh plot evidence and render all figures without touching originals."""
    FIG.mkdir(exist_ok=True)
    if any(FIG.glob('fig*')):raise FileExistsError('Figure outputs already exist')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
        'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42})
    data=core.load_all_karakoca(ROOT/'Datasets/Sub_THz_measurements')
    d,a,pl,ds,bc=core.extract_statistics(data,angle_cap=20)
    X,D,A=core.make_2d_features(d,a);y=pl.ravel()
    wl,wa=core.fit_separable(d,a,pl);wp=core.fit_poly2d(X,y)
    mlp=core.build_mlp();mlp.fit(X,y)
    gpr=GaussianProcessRegressor(kernel=grouped.make_gpr_kernel(),alpha=0,
        n_restarts_optimizer=5,normalize_y=True,random_state=42).fit(X,y)
    print('Full fits done; generating LOO plot evidence',flush=True)
    cv=[core.loo_cv_separable(d,a,pl),core.loo_cv_generic(X,y,'poly2d'),core.loo_cv_generic(X,y,'mlp')]
    print('M1-M3 LOO done',flush=True)
    mu,std=np.zeros(len(y)),np.zeros(len(y))
    for i in range(len(y)):
        mask=np.arange(len(y))!=i
        g=GaussianProcessRegressor(kernel=grouped.make_gpr_kernel(),alpha=0,
            n_restarts_optimizer=2,normalize_y=True,random_state=42).fit(X[mask],y[mask])
        yp,sp=g.predict(X[i:i+1],return_std=True);mu[i],std[i]=yp[0],sp[0]
        if (i+1)%30==0:print('M4 LOO',i+1,flush=True)
    cv.append(core._cv_metrics(y,mu))
    preds=pd.DataFrame({'distance_m':D,'angle_deg':A,'measured_pl_db':y,'M4_std_db':std})
    for name,result in zip(NAMES,cv):preds[name+'_prediction_db']=result['y_pred']
    csv_new(preds,'figure_loo_predictions.csv')
    csv_new(pd.DataFrame([dict(model=nm,rmse_db=c['rmse'],mae_db=c['mae'],r2=c['r2']) for nm,c in zip(NAMES,cv)]),'figure_loo_metrics.csv')
    half=1.645*std
    json_new(dict(m4_mean_full_width_db=float(2*half.mean()),m4_mean_half_width_db=float(half.mean()),
        m4_median_half_width_db=float(np.median(half)),m4_pointwise_coverage=float(np.mean(np.abs(mu-y)<=half)),
        m4_legacy_constant_halfwidth_coverage=float(np.mean(np.abs(mu-y)<=half.mean())),
        m4_full_fit_kernel=str(gpr.kernel_),m4_grid_transformation='[log10(d/0.20), theta/20]'), 'prediction_interval_audit.json')
    tek=core.load_tekbiyik(ROOT/'Datasets/THz_Wireless_Measurements/THz_measurement_channel_frequency_response.mat')
    td=tek['distances_m'];ty=-tek['cfr_db'].mean(axis=0)
    tx=core.make_2d_features(td,np.array([0]))[0]
    tp=[core.sep_predict(td,0,wl,wa),core.poly2d_predict(tx,wp),mlp.predict(tx),gpr.predict(tx)]
    ext=pd.DataFrame({'distance_m':td,'measured_pl_db':ty})
    for nm,p in zip(NAMES,tp):ext[nm+'_prediction_db']=p
    csv_new(ext,'external_validation_predictions.csv')
    selected=[0,5,10,15,20]
    fig,axes=newfig(height=3.35);ax=axes[0,0]
    for theta,c in zip(selected,plt.cm.viridis(np.linspace(0,.95,5))):ax.plot(d*100,pl[:,np.where(a==theta)[0][0]],'o-',ms=3,color=c,label=f'{theta}°')
    ax.plot(td*100,ty,'ks--',ms=3,label='Tekbiyik, 0°')
    # The original figure uses a gain-adjusted FSPL overlay; retain its explicit convention.
    fspl=core.fspl_db(d);offset=float(np.mean(fspl-pl[:,0]))
    ax.plot(d*100,fspl-offset,'k-.',lw=1,label='FSPL, fitted offset')
    ax.set(xlabel='Distance (cm)',ylabel='Band-average path loss (dB)');ax.legend(ncol=2,loc='upper left')
    save(fig,1,'primary/secondary CFR; theoretical n=2 with fitted system offset')
    fig,axes=newfig();ax=axes[0,0]
    ax.plot(d*100,pl[:,0]-pl[0,0],'o',ms=4,label='Karakoca')
    ax.plot(td*100,ty-ty[0],'s',ms=4,label='Tekbiyik')
    ax.plot(d*100,wl[0]*np.log10(d/.2),'r-',label=f'Fitted slope n={wl[0]/10:.3f}')
    ax.plot(d*100,20*np.log10(d/.2),'k--',label='Free-space slope n=2')
    ax.set(xlabel='Distance (cm)',ylabel='Relative path loss (dB)');ax.legend()
    save(fig,2,'measured LoS normalized separately at 20 cm; fitted slope curve (not absolute CI anchor)')
    fig,axes=newfig();ax=axes[0,0]
    for i in [0,2,4,6,8]:ax.plot(a,pl[i]-pl[i,0],'o-',ms=3,label=f'{d[i]*100:.0f} cm')
    ax.plot(a,wa[0]*a*a+wa[1]*a,'k--',label='M1 angular fit');ax.axvspan(8,13,color='gray',alpha=.1)
    ax.set(xlabel='Misalignment angle (deg)',ylabel='Angular penalty (dB)');ax.legend(ncol=2)
    save(fig,3,'primary CFR angular penalties and M1 fit')
    fig,axes=newfig(1,2,height=2.95)
    for theta in selected:
        j=np.where(a==theta)[0][0];axes[0,0].semilogy(d*100,bc[:,j],'o-',ms=3,label=f'{theta}°')
    axes[0,0].axhline(60000,color='k',ls='--',label='60 GHz band');axes[0,0].legend(ncol=2)
    axes[0,0].set(xlabel='Distance (cm)',ylabel='System coherence BW (MHz)')
    axes[0,1].semilogy(d*100,bc[:,0],'o-',label='0°');axes[0,1].semilogy(d*100,bc[:,-1],'s-',label='20°');axes[0,1].legend()
    axes[0,1].set(xlabel='Distance (cm)',ylabel='System coherence BW (MHz)')
    save(fig,4,'core CFR amplitude-autocorrelation estimator; no change to numerical definition')
    fig,axes=newfig(1,2,height=2.95)
    for theta in selected:axes[0,0].plot(d*100,ds[:,np.where(a==theta)[0][0]]*1000,'o-',ms=3,label=f'{theta}°')
    for i in [0,2,4,5,6,8]:axes[0,1].plot(a,ds[i]*1000,'o-',ms=3,label=f'{d[i]*100:.0f} cm')
    axes[0,0].set(xlabel='Distance (cm)',ylabel='RMS delay spread (ps)');axes[0,1].set(xlabel='Misalignment angle (deg)',ylabel='RMS delay spread (ps)')
    for ax in axes.flat:ax.legend(ncol=2)
    save(fig,5,'provided CIR; core causal-half and -25 dB threshold; outlier shown, not removed')
    penalty=pl-pl[:,0:1];deviation=penalty-penalty.mean(axis=0);peak=np.max(np.abs(deviation),axis=0)
    fig,axes=newfig(1,2,height=3.1)
    im=axes[0,0].pcolormesh(d*100,a,penalty.T,shading='nearest',cmap='plasma');fig.colorbar(im,ax=axes[0,0],label='Angular penalty (dB)')
    axes[0,0].set(xlabel='Distance (cm)',ylabel='Misalignment angle (deg)')
    axes[0,1].bar(a,peak,color=COLORS[0]);axes[0,1].axvspan(8,13,color='gray',alpha=.1)
    axes[0,1].set(xlabel='Misalignment angle (deg)',ylabel='Max centered penalty (dB)')
    save(fig,6,'actual nonuniform angle coordinates; centered cross-distance penalty; no unverified noise threshold')
    dense=np.linspace(.2,1,200)
    fig,axes=newfig(2,2,height=5.05)
    for ax,theta in zip(axes.flat,[0,5,10,20]):
        xx=core.make_2d_features(dense,np.array([theta]))[0];gm,gs=gpr.predict(xx,return_std=True)
        ax.plot(d*100,pl[:,np.where(a==theta)[0][0]],'ko',ms=3,label='Measured')
        ax.plot(dense*100,core.sep_predict(dense,theta,wl,wa),color=COLORS[0],label='M1')
        ax.plot(dense*100,core.poly2d_predict(xx,wp),'--',color=COLORS[1],label='M2')
        ax.plot(dense*100,gm,color=COLORS[3],label='M4');ax.fill_between(dense*100,gm-1.645*gs,gm+1.645*gs,color=COLORS[3],alpha=.18)
        ax.set(xlabel='Distance (cm)',ylabel='Path loss (dB)',title=f'{theta}°');ax.legend(ncol=2)
    save(fig,7,'fresh full fits; dense correctly transformed grid; pointwise model 90% PI')
    limits=[y.min()-1,y.max()+1]
    fig,axes=newfig(2,2,height=5.1)
    for ax,nm,c,col in zip(axes.flat,NAMES,cv,COLORS):
        ax.scatter(y,c['y_pred'],s=9,color=col);ax.plot(limits,limits,'k--',lw=1)
        ax.set(xlabel='Measured PL (dB)',ylabel='LOO prediction (dB)',title=nm,xlim=limits,ylim=limits)
    save(fig,8,'fresh pointwise LOO predictions; 2x2 at two-column width')
    fig,axes=newfig(height=3.2);ax=axes[0,0]
    sc=ax.scatter(y,mu,c=std,s=12,cmap='plasma');ax.plot(limits,limits,'k--',lw=1)
    fig.colorbar(sc,ax=ax,label='Predictive std (dB)');ax.set(xlabel='Measured PL (dB)',ylabel='M4 LOO prediction (dB)')
    save(fig,9,'fresh M4 pointwise LOO prediction and std')
    fig,axes=newfig(2,2,height=4.9)
    for ax,nm,c,col in zip(axes.flat,NAMES,cv,COLORS):
        residual=c['y_pred']-y;ax.hist(residual,bins=16,color=col,edgecolor='white');ax.axvline(0,color='k',ls='--')
        ax.set(xlabel='Prediction − measured (dB)',ylabel='Count',title=nm);ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig,10,'all four LOO residuals; individual axis ranges expose distributions; 2x2')
    dg,ag=np.meshgrid(np.linspace(.2,1,100),np.linspace(0,20,100))
    xg=np.column_stack([np.log10(dg.ravel()/.2),ag.ravel()/20])
    gm,gs=gpr.predict(xg,return_std=True)
    csv_new(pd.DataFrame({'distance_m':dg.ravel(),'angle_deg':ag.ravel(),'M4_mean_db':gm,'M4_std_db':gs}),'figure_surface_predictions.csv')
    # Match the training transform exactly at measured coordinates.
    np.testing.assert_allclose(gpr.predict(np.column_stack([np.log10(D/.2),A/20])),gpr.predict(X),atol=1e-12)
    fig,axes=newfig(1,2,height=3.15)
    for ax,value,label in zip(axes.flat,[gm,gs],['Mean path loss (dB)','Predictive std (dB)']):
        im=ax.contourf(dg*100,ag,value.reshape(dg.shape),levels=20,cmap='plasma');fig.colorbar(im,ax=ax,label=label)
        ax.scatter(D*100,A,s=3,color='white');ax.set(xlabel='Distance (cm)',ylabel='Misalignment angle (deg)')
    save(fig,11,'corrected theta/20 surface features, replacing inconsistent old raw-degree evaluation')
    fig,axes=newfig(height=3.2);ax=axes[0,0]
    im=ax.contourf(dg*100,ag,core.sep_predict(dg,ag,wl,wa),levels=20,cmap='plasma');fig.colorbar(im,ax=ax,label='Path loss (dB)')
    ax.scatter(D*100,A,s=3,color='white');ax.set(xlabel='Distance (cm)',ylabel='Misalignment angle (deg)')
    save(fig,12,'M1 coefficients freshly fitted; same parameterization as frozen artifacts')
    mat=sio.loadmat(ROOT/'SJTU_Global_Results.mat',squeeze_me=True);values=list(mat['n_vals'][:3])+[wl[0]/10,float(mat['n_kim'])]
    labels=['Corridor (sim.)','Hallway (sim.)','Atrium (sim.)','Anechoic (meas.)','Street (meas.)'];order=np.argsort(values)
    fig,axes=newfig(height=3.15);ax=axes[0,0]
    ax.barh(np.arange(5),np.array(values)[order],color='#4477AA')
    ax.set_yticks(np.arange(5),np.array(labels)[order]);ax.axvline(2,color='k',ls='--');ax.set(xlabel='Reported path-loss exponent n',xlim=(1.3,2.16))
    for j,i in enumerate(order):ax.text(values[i]+.01,j,f'{values[i]:.3f}',va='center',fontsize=9)
    save(fig,13,'frozen SJTU summary and primary floating-intercept fit; heterogeneous fit conventions acknowledged')
    json_new({'figures':AUDIT},'figure_generation_manifest.json')
    print('13 vector PDFs and 13 600-dpi PNGs generated',flush=True)


if __name__=='__main__':main()
