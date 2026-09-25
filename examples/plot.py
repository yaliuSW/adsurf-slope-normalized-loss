"""Plot saved examples or a newly completed paired inversion."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
COLORS={'raw':'#0072B2','retained':'#D55E00'}


def profile(ax,vs,h,end,**kwargs):
    top=np.r_[0,np.cumsum(h[:-1])]
    bottom=np.r_[top[1:],end]
    ax.plot(np.repeat(vs,2),np.column_stack([top,bottom]).ravel(),**kwargs)


def plot_case(case,output,run=None):
    arrays=np.load(HERE/'data'/(case+'.npz'))
    data=Path(run)/'selection' if run else HERE/'results'/case
    models={m:np.load(data/(m+'_selected.npz')) for m in ['raw','retained']}
    shallow=case=='field' or case=='strong_lvz' or case.startswith('noise_')
    unit=1000 if shallow else 1
    end=(80 if case=='field' else 25) if shallow else (2.9 if case=='gradient' else 12.2)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    ax=axes[0]
    for sid in np.linspace(0,len(arrays['starts_vs'])-1,min(12,len(arrays['starts_vs']))).astype(int):
        profile(ax,arrays['starts_vs'][sid]*unit,arrays['starts_h'][sid]*unit,end,color='.85',lw=.6)
    if 'true_vs' in arrays and arrays['true_vs'].size:
        profile(ax,arrays['true_vs']*unit,arrays['true_h']*unit,end,color='black',lw=1.4,label='True')
    elif case=='field':
        bh=np.genfromtxt(HERE/'data/borehole_reference.csv',delimiter=',',names=True)
        ax.plot(bh['panel_a_Vs_m_s'],bh['panel_a_depth_m'],color='black',lw=1.2,label='Borehole reference')
    for method,label in [('retained','SN-ADsurf'),('raw','ADsurf')]:
        model=models[method]
        profile(ax,model['vs']*unit,model['h']*unit,end,color=COLORS[method],
                lw=2 if method=='retained' else 1.2,ls='-' if method=='retained' else '--',label=label)
    ax.set(xlabel=f'Shear-wave velocity ({"m/s" if shallow else "km/s"})',
           ylabel=f'Depth ({"m" if shallow else "km"})',ylim=(end,0),title='a  Selected velocity profiles')
    ax.legend(frameon=False,fontsize=8)
    obs=arrays['obs']
    for g in np.unique(obs[:,2]):
        mask=obs[:,2]==g;order=np.argsort(obs[mask,0])
        axes[1].scatter(obs[mask,0],obs[mask,1]*unit,s=9,facecolors='none',edgecolors='.25',
                        label='Observed groups' if g==0 else None)
        for method,label in [('retained','SN-ADsurf'),('raw','ADsurf')]:
            axes[1].plot(obs[mask,0][order],models[method]['pred'][mask][order]*unit,
                         color=COLORS[method],ls='-' if method=='retained' else '--',
                         lw=1.8 if method=='retained' else 1.1,label=label if g==0 else None)
    axes[1].set(xlabel='Frequency (Hz)',ylabel=f'Phase velocity ({"m/s" if shallow else "km/s"})',title='b  Dispersion fit')
    axes[1].legend(frameon=False,fontsize=8)
    for method,label,color in [('raw','ADsurf',COLORS['raw']),('sn','Switching path',COLORS['retained'])]:
        path=Path(run)/method/'trajectory.npz' if run else HERE/'results'/case/(method+'_trajectory.npz')
        if path.exists():
            tr=np.load(path);loss=tr['training_loss']
            axes[2].semilogy(np.arange(1,len(loss)+1),np.median(loss,axis=1),color=color,label=label)
    axes[2].set(xlabel='Update',ylabel='Median path objective',title='c  Optimization history')
    if axes[2].lines:axes[2].legend(frameon=False,fontsize=8)
    else:axes[2].text(.5,.5,'Run the example to plot its history',ha='center',va='center',transform=axes[2].transAxes)
    fig.suptitle(case.replace('_',' ').capitalize())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(output/(case+'.'+suffix),dpi=200)
    plt.close(fig)


def plot_noise(output):
    groups=['iid_gaussian','frequency_correlated','mode_bias','sparse_outlier']
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),layout='constrained',sharey=True)
    for ax,group in zip(axes,groups):
        for folder in sorted((HERE/'results').glob('noise_'+group+'_seed_*')):
            rows={r['method']:r for r in json.loads((folder/'summary.json').read_text())}
            values=[rows[m]['Vs_RMSE_m_s'] for m in ['raw','retained']]
            ax.plot([0,1],values,color='.65',lw=.8)
            ax.scatter([0,1],values,c=[COLORS['raw'],COLORS['retained']],s=20,zorder=3)
        ax.set(xticks=[0,1],xticklabels=['ADsurf','SN-ADsurf'],title=group.replace('_',' ').capitalize())
    axes[0].set_ylabel('Shear-wave velocity RMSE (m/s)')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(output/('noise_comparison.'+suffix),dpi=200)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='field');parser.add_argument('--out',default='plots')
    parser.add_argument('--run',help='New run directory containing selection/, raw/ and sn/')
    parser.add_argument('--noise',action='store_true')
    args=parser.parse_args()
    if args.noise:plot_noise(args.out)
    else:plot_case(args.case,args.out,args.run)
