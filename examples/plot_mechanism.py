"""Plot the saved local-slope experiment without rerunning inversion."""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot import profile


def main(output):
    data=Path(__file__).resolve().parent/'data/local_slope'
    bank=np.load(data/'curves.npz')
    rows=list(csv.DictReader((data/'metrics.csv').open()))
    rmse=np.array([float(r['physical_RMSE_m_s']) if r['physical_RMSE_m_s'] else np.nan for r in rows])
    valid=np.isfinite(rmse)
    assert len(rows)==500 and valid.sum()==349
    colors={'A':'#0072B2','B':'#D55E00'}
    fig,axes=plt.subplots(2,2,figsize=(8,6),layout='constrained')
    a=axes.ravel()
    for vs in bank['vs']:profile(a[0],vs*1000,bank['h']*1000,20,color='.86',lw=.4)
    profile(a[0],bank['true_vs']*1000,bank['h']*1000,20,color='black',lw=1.3,label='True')
    obs=bank['obs']
    for label,sid in [('A',181),('B',438)]:
        profile(a[0],bank['vs'][sid]*1000,bank['h']*1000,20,color=colors[label],lw=1.2,label=label)
        for g in np.unique(obs[:,2]):
            mask=obs[:,2]==g;order=np.argsort(obs[mask,0])
            a[1].plot(obs[mask,0][order],bank[label][mask][order]*1000,color=colors[label],label=label if g==0 else None)
    a[0].set(xlabel='Shear-wave velocity (m/s)',ylabel='Depth (m)',ylim=(20,0),title='a  Fixed candidate models')
    a[0].legend(frameon=False,fontsize=8)
    a[1].scatter(obs[:,0],obs[:,1]*1000,s=10,facecolors='none',edgecolors='.3')
    a[1].set(xlabel='Frequency (Hz)',ylabel='Phase velocity (m/s)',title='b  Two candidates')
    a[1].legend(frameon=False,fontsize=8)
    display_scale=float(rows[0]['raw_group_L1'])/float(rows[0]['alpha1_group_L1'])
    for key,label,color,scale in [('raw_group_L1','Raw','#0072B2',1),('alpha1_group_L1','SN','#D55E00',display_scale)]:
        values=np.array([float(r[key]) for r in rows])*scale
        a[2].scatter(rmse[valid],values[valid],s=7,c=color,alpha=.5,label=label)
        for sid,name in [(181,'A'),(438,'B')]:
            a[2].scatter(rmse[sid],values[sid],s=30,facecolors='white',edgecolors=color,zorder=4)
            a[2].annotate(name,(rmse[sid],values[sid]),xytext=(6,10 if label=='Raw' else -14),
                          textcoords='offset points',color=color,fontsize=9)
    a[2].set(xlabel='All-point dispersion RMSE (m/s)',ylabel='Objective (SN display-scaled)',
             xlim=(0,40),yscale='log',title='c  Residual objectives')
    a[2].legend(frameon=False,fontsize=8)
    corrected=np.array([float(r['EPVR_RMS_m_s']) for r in rows])
    a[3].scatter(rmse[valid],corrected[valid],s=7,c='.5',alpha=.5)
    a[3].plot([0,40],[0,40],'k--',lw=.8)
    a[3].set(xlabel='All-point dispersion RMSE (m/s)',ylabel='Slope-corrected RMS (m/s)',
             xlim=(0,40),ylim=(0,40),title='d  Local correction')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf']:fig.savefig(output/('local_slope.'+extension),dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='plots')
    main(parser.parse_args().out)
