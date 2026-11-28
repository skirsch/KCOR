"""Verify the matched study's direct fitting route against ordinary workbook output."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import misspecification_study as M
ROOT=Path(sys.argv[1]);M.init(ROOT)
sys.path.insert(0,str(ROOT/'test'))
from test_round7_workbook import run,K
dates=pd.date_range('2021-06-14',periods=M.W,freq='7D')
a,b=dates[20],dates[80]
K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE=[dict(start_iso=a.strftime('%G-%V'),end_iso=b.strftime('%G-%V'),start_int=int(a.strftime('%G%V')),end_int=int(b.strftime('%G%V')),label='matched study')]
results=[]
for i,scenario in enumerate(M.SCENARIOS):
 rng=np.random.default_rng(np.random.SeedSequence([M.SEED,i,0,0]))
 if scenario=='correlated_gamma':orig=[M.cluster_cells(rng,th).sum(axis=0) for th in M.THETA]
 else:orig=[rng.multinomial(M.N,M.probabilities(scenario,th)) for th in M.THETA]
 v,_,_=M.pair(*orig,point=True)
 frames=[]
 for dose,c in zip([1,0],orig):
  frames.append(pd.DataFrame(dict(YearOfBirth=1950,Dose=dose,Sex='O',DateDied=dates,ISOweekDied=dates.strftime('%G-%V'),Alive=np.cumsum(c[::-1])[::-1][:-1],Dead=c[:-1])))
 out=run('2021_24',pd.concat(frames,ignore_index=True))
 k=float(out[out.YearOfBirth.eq(1950)].KCOR.iloc[-1]);diff=abs(k-v)
 print(scenario,v,k,diff,flush=True);assert diff<1e-8
 results.append(dict(scenario=scenario,direct=v,workbook=k,max_diff=diff))
(Path(__file__).parent/'workbook-verification.json').write_text(json.dumps(results,indent=2))
