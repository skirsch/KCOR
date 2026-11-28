"""Fixed expected-deaths weights; preserve original empirical weighting."""
import numpy as np
def expected_deaths_weights(rates, exposures, eps=1e-12):
    rates=[np.asarray(a,float) for a in rates]
    all_positive=np.concatenate([a[a>eps] for a in rates]) if rates else np.array([])
    fallback=float(all_positive.min()) if all_positive.size else eps
    weights=np.array([float(a[a>eps].mean() if (a>eps).any() else fallback)*float(pt) for a,pt in zip(rates,exposures)])
    return weights/weights.sum() if weights.sum()>0 else np.full(len(weights),1/len(weights)) if len(weights) else weights
