import argparse,math,sys,warnings
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument("candidate",type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.candidate/"code"))
import KCOR as K
with warnings.catch_warnings(record=True) as flags:
    warnings.simplefilter("always")
    hazard=K.hazard_from_mr([0,.001,.9999,1,1.1,-.1])
    np.testing.assert_allclose(hazard[:3],[0,-math.log1p(-.001),-math.log1p(-.9999)])
    assert np.isinf(hazard[3:]).all()
    short=K.invert_gamma_frailty([.01],90)[0]
    extended=K.invert_gamma_frailty([.01,2,7.9,9],90)
    assert short==extended[0]
    assert np.isfinite(extended[:3]).all() and np.isnan(extended[3])
    assert np.isnan(K.invert_gamma_frailty([.01,np.inf],90)[1])
    assert np.isnan(K.invert_gamma_frailty([.01,np.inf],0)[1])
    assert len(flags)>=3
    print("hazards",hazard,"inversions",extended,"warnings",len(flags))
