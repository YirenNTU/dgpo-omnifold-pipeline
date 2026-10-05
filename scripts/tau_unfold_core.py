"""Small NumPy/RooUnfold interface; no inference or training dependencies."""
import uuid
import numpy as np


def nested_sets(group, response, fractions, seed):
    """Stratified nested subsets; never draws events outside response pool."""
    if any(not 0<f<=1 for f in fractions): raise ValueError('Invalid response fraction')
    rng=np.random.default_rng(seed); selected={f:[] for f in fractions}
    for g in np.unique(group[response]):
        ii=response[group[response]==g].copy();rng.shuffle(ii)
        for f in fractions: selected[f].extend(ii[:max(1,int(len(ii)*f))])
    return {f:np.sort(v) for f,v in selected.items()}


def summarize_pseudoexperiments(estimates, sigmas, truth, expected):
    estimates,sigmas=np.asarray(estimates,float),np.asarray(sigmas,float)
    if estimates.shape!=sigmas.shape or estimates.ndim!=1 or len(estimates)<2 or not np.isfinite(estimates).all() or not np.isfinite(sigmas).all() or (sigmas<=0).any():
        raise ValueError('Need finite pseudoexperiments with positive uncertainties')
    pulls=(estimates-truth)/sigmas
    centered=(estimates-expected)/sigmas
    coverage=float(np.mean(np.abs(pulls)<=1))
    return dict(mean=float(estimates.mean()),mean_residual=float(estimates.mean()-truth),
        mean_mc_se=float(estimates.std(ddof=1)/np.sqrt(len(estimates))),
        empirical_std=float(estimates.std(ddof=1)),mean_sigma=float(sigmas.mean()),
        std_over_mean_sigma=float(estimates.std(ddof=1)/sigmas.mean()),
        pull_mean=float(pulls.mean()),pull_std=float(pulls.std(ddof=1)),
        coverage68=coverage,coverage95=float(np.mean(np.abs(pulls)<=1.959963984540054)),
        coverage68_mc_se=float(np.sqrt(coverage*(1-coverage)/len(estimates))),
        centered_pull_mean=float(centered.mean()),centered_pull_std=float(centered.std(ddof=1)),
        centered_coverage68=float(np.mean(np.abs(centered)<=1)))


def indices(u, bins):
    u = np.asarray(u, float)
    if not np.isfinite(u).all() or np.max(np.abs(u)) > 1+1e-8:
        raise ValueError('Invalid angular products')
    edges = np.linspace(-1,1,int(bins)+1) if np.ndim(bins)==0 else np.asarray(bins,float)
    return np.clip(np.searchsorted(edges,u,side='right')-1,0,len(edges)-2)


def response_edges(products, groups, weights, initial_bins=10, min_count=20):
    """Common edges from response events ONLY; merge sparse adjacent bins.

    Every nominal truth/reco histogram in both arms must have enough positive
    weight events AND effective entries. No test events or target labels enter.
    """
    w=np.asarray(weights,float); groups=np.asarray(groups)
    if min_count<1: raise ValueError('min_count must be positive')
    edges=np.linspace(-1,1,initial_bins+1)
    def inventory(edges):
        records=[]
        for name,values in products.items():
            ids=indices(values,edges)
            for g in np.unique(groups):
                take=(groups==g)&(w>0)
                count=np.bincount(ids[take],minlength=len(edges)-1)
                mass=np.bincount(ids[take],weights=w[take],minlength=len(edges)-1)
                square=np.bincount(ids[take],weights=w[take]**2,minlength=len(edges)-1)
                ess=np.divide(mass**2,square,out=np.zeros_like(mass),where=square>0)
                records.append(dict(source=name,group=int(g),count=count.tolist(),effective_entries=ess.tolist()))
        return records
    original=inventory(edges); merges=[]
    while True:
        current=inventory(edges)
        quality=np.array([np.minimum(r['count'],r['effective_entries']) for r in current])
        if quality.min()>=min_count: break
        row,bad=np.unravel_index(quality.argmin(),quality.shape)
        if len(edges)-1<=2:
            raise ValueError(f'Insufficient response support even at two bins: {current[row]}; need more response data')
        # Remove the adjacent boundary producing the narrower combined bin.
        candidates=[]
        if bad>0: candidates.append((edges[bad+1]-edges[bad-1],bad))
        if bad<len(edges)-2: candidates.append((edges[bad+2]-edges[bad],bad+1))
        _,boundary=min(candidates)
        merges.append(dict(removed_edge=float(edges[boundary]),source=current[row]['source'],
                           group=current[row]['group'],sparse_bin=int(bad),effective_count=float(quality[row,bad])))
        edges=np.delete(edges,boundary)
    return edges,dict(initial=original,final=current,merges=merges,edges=edges.tolist())


def moment(counts, covariance, coefficients):
    counts, covariance, coefficients = map(np.asarray, (counts, covariance, coefficients))
    total = counts.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Nonpositive unfolded normalization')
    value = float(counts@coefficients/total)
    gradient = (coefficients-value)/total
    variance = float(gradient@covariance@gradient)
    if variance < -1e-9 or not np.isfinite(variance):
        raise ValueError('Invalid propagated variance')
    return value, float(np.sqrt(max(variance, 0)))


class SVDResponse:
    def __init__(self, ROOT, truth_bins, reco_bins, weights, bins):
        self.ROOT, self.bins = ROOT, bins
        self.response = ROOT.RooUnfoldResponse(bins, -.5, bins-.5)
        mass = np.bincount(truth_bins, weights=weights, minlength=bins)
        if np.any(mass <= 0):
            raise ValueError('Empty truth response bin; reduce bins or increase response data')
        for t, r, w in zip(truth_bins, reco_bins, weights):
            if w > 0:
                self.response.Fill(float(r), float(t), float(w))

    def unfold(self, counts, k, method='svd'):
        R, n = self.ROOT, self.bins
        hist = R.TH1D('measured_'+uuid.uuid4().hex, '', n, -.5, n-.5)
        hist.SetDirectory(0)
        for i, v in enumerate(counts):
            hist.SetBinContent(i+1, float(v))
            hist.SetBinError(i+1, float(np.sqrt(v)))
        if method == 'svd':
            obj = R.RooUnfoldSvd(self.response, hist, k)
        elif method == 'invert':
            obj = R.RooUnfoldInvert(self.response, hist)
        else:
            raise ValueError('Unknown unfolding method')
        obj.SetVerbose(0)
        obj.IncludeSystematics(0)  # response-MC variation is bootstrapped separately
        # RooUnfold kCovariance=2: retain full bin covariance, not diagonal errors.
        out = obj.Hunfold(2) if hasattr(obj,'Hunfold') else obj.Hreco(2)
        cov = obj.Eunfold(2) if hasattr(obj,'Eunfold') else obj.Ereco(2)
        values = np.array([out.GetBinContent(i+1) for i in range(n)])
        matrix = np.array([[cov[i][j] for j in range(n)] for i in range(n)])
        if not np.isfinite(values).all() or not np.isfinite(matrix).all():
            raise ValueError('Nonfinite RooUnfold output')
        return values, matrix
