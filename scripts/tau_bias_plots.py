"""Direct observable plots; intervals are fixed-panel resampling variability."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scripts.tau_bias_sampling import AXES


def direct_plots(report, output):
    paths = {}
    cases = [c for c in report['cases'] if 'C' in c]
    colors = {'truth':'black','pretrain':'tab:blue','dgpo':'tab:orange'}
    for ref in ('full','matched'):
        # Compare signed C values directly, not absolute errors or delta C.
        fig=Figure(figsize=(12,4),layout='constrained'); FigureCanvasAgg(fig)
        matrices=[np.asarray(report['nominal_full_panel_C'][key])
                  for key in ('truth/'+ref,'pretrain','dgpo')]
        limit=max(float(np.max(np.abs(m))) for m in matrices)
        limit=max(limit,.01)
        panels=fig.subplots(1,3)
        for ax,matrix,label in zip(panels,matrices,('Truth','Pretrain','DGPO')):
            heat=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set(title=label,xlabel='b axis',ylabel='a axis')
            ax.set_xticks(range(3),list('krn')); ax.set_yticks(range(3),list('krn'))
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',
                            color='white' if abs(matrix[i,j])>.6*limit else 'black')
        fig.colorbar(heat,ax=list(panels),label='Signed Cij value',shrink=.8)
        fig.suptitle(f'Cij actual matrix values | nominal full panel | {ref} truth reference')
        path=output/f'C_matrix_values_{ref}.png'; fig.savefig(path,dpi=140)
        paths[f'bias/direct/C_matrix_values/{ref}']=path

        # Only each component's own injection cases, plus nominal.
        fig = Figure(figsize=(13,11),layout='constrained'); FigureCanvasAgg(fig)
        for j,ax in enumerate(fig.subplots(3,3).ravel()):
            subset = [cases[0]]+[c for c in cases[1:] if c['case'].split('/')[0]==AXES[j]]
            x = np.array([np.asarray(c['C']['truth/'+ref]).ravel()[j] for c in subset])
            bounds = list(x)
            for arm in ('pretrain','dgpo'):
                y = np.array([np.asarray(c['C'][arm]).ravel()[j] for c in subset])
                for i,c in enumerate(subset):
                    xl,xh = np.asarray(c['C_interval95']['truth/'+ref])[:,j]
                    yl,yh = np.asarray(c['C_interval95'][arm])[:,j]
                    ax.plot([xl,xh],[y[i],y[i]],color=colors[arm],alpha=.3)
                    ax.plot([x[i],x[i]],[yl,yh],color=colors[arm],alpha=.3)
                    bounds.extend([xl,xh,yl,yh])
                order = np.argsort(x)
                ax.plot(x[order],y[order],'o-',color=colors[arm],label=arm)
                ax.scatter(x[0],y[0],marker='*',s=110,color=colors[arm],zorder=4)
                flagged=[i for i,c in enumerate(subset) if c.get('concentration_warning')]
                ax.scatter(x[flagged],y[flagged],marker='x',s=80,color='red',zorder=5)
            lo,hi = min(bounds),max(bounds); pad=max((hi-lo)*.08,.01)
            ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'k--',lw=.8,label='ideal y=x')
            ax.set(title='C'+AXES[j],xlabel='Truth C'+AXES[j],ylabel='Generated C'+AXES[j])
            ax.legend(fontsize=7)
        fig.suptitle(f'Cij actual values | {ref} | nominal: star; low support: red cross\n95% resampling ranges, not error on the mean; no fitted calibration')
        path=output/f'C_actual_{ref}.png'; fig.savefig(path,dpi=140); paths[f'bias/direct/C/{ref}']=path

        fig=Figure(figsize=(13,10),layout='constrained'); FigureCanvasAgg(fig)
        for j,ax in enumerate(fig.subplots(3,3).ravel()):
            subset=[cases[0]]+[c for c in cases[1:] if c['case'].split('/')[0]==AXES[j]]
            x=np.array([np.asarray(c['C']['truth/'+ref]).ravel()[j] for c in subset]); order=np.argsort(x)
            for arm in ('pretrain','dgpo'):
                means=np.array([c['C_bias'][ref+'/'+arm]['mean'][j] for c in subset])
                bands=np.array([c['C_bias'][ref+'/'+arm]['interval95'] for c in subset])[:,:,j]
                ax.plot(x[order],means[order],'o-',label=arm,color=colors[arm])
                ax.fill_between(x[order],bands[order,0],bands[order,1],alpha=.12,color=colors[arm])
            ax.axhline(0,color='black',ls='--',lw=.8)
            ax.set(title='C'+AXES[j],xlabel='Actual truth C',ylabel='Generated - truth'); ax.legend(fontsize=7)
        fig.suptitle(f'Signed moment bias | {ref} | paired 95% resampling ranges')
        path=output/f'C_bias_{ref}.png'; fig.savefig(path,dpi=140); paths[f'bias/direct/signed_bias/{ref}']=path

        fig=Figure(figsize=(14,9),layout='constrained'); FigureCanvasAgg(fig)
        for j,ax in enumerate(fig.subplots(2,3).ravel()):
            for key,label in [('truth/'+ref,'truth'),('pretrain','pretrain'),('dgpo','dgpo')]:
                y=np.array([c['B'][key][j] for c in cases])
                interval=np.array([c['B_interval95'][key] for c in cases])
                ax.plot(range(len(cases)),y,'o',ms=3,label=label,color=colors[label])
                ax.fill_between(range(len(cases)),interval[:,0,j],interval[:,1,j],color=colors[label],alpha=.1)
            ax.set(title='B '+('a' if j<3 else 'b')+'_'+'krn'[j%3],ylabel='Signed-kappa angular moment')
            ax.set_xticks(range(len(cases)),[c['case'] for c in cases],rotation=90,fontsize=6)
            ax.legend(fontsize=7)
        fig.suptitle(f'B angular moments | {ref} | all C-product sampling scenarios\nNot a dedicated B injection scan; not acceptance-corrected polarization')
        path=output/f'B_actual_{ref}.png'; fig.savefig(path,dpi=140); paths[f'bias/direct/B/{ref}']=path

        fig=Figure(figsize=(12,4),layout='constrained'); FigureCanvasAgg(fig)
        for ax,field,labels in zip(fig.subplots(1,2),('C','B'),(AXES,['a_'+i for i in 'krn']+['b_'+i for i in 'krn'])):
            for key,label in [('truth/'+ref,'truth'),('pretrain','pretrain'),('dgpo','dgpo')]:
                vals=np.asarray(report['nominal_full_panel_'+field][key]).ravel()
                ax.plot(range(len(vals)),vals,'o-',label=label,color=colors[label])
            ax.set_xticks(range(len(labels)),labels); ax.set_title(field+' nominal values'); ax.legend()
        fig.suptitle(f'Nominal panel | {ref} | B uses signed-kappa moment convention')
        path=output/f'nominal_{ref}.png'; fig.savefig(path,dpi=140); paths[f'bias/direct/nominal/{ref}']=path
    return paths
