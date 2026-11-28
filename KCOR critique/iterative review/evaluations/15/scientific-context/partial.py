from pathlib import Path
import sys, os
import numpy as np,pandas as pd
root=Path(sys.argv[1]).resolve();sys.path[:0]=[str(root/'code'),str(root/'test')]
from test_round14_aggregate import written,input_frame,total,ending
if __name__=='__main__':
    f=input_frame('2023_01');f=f[f.YearOfBirth.eq(1940)].copy();g=total(f).assign(YearOfBirth=-60)
    ix=g.index[g.Dose.eq(1)];g.loc[ix,'Dead']*=np.where(np.arange(len(ix))>5,4,1);g.loc[ix,'Alive']=40000-np.r_[0,g.loc[ix,'Dead'].to_numpy().cumsum()[:-1]]
    out=written(pd.concat([f,g],ignore_index=True),'partial')
    print(out.groupby('YearOfBirth').tail(1)[['YearOfBirth','KCOR','normalization_status','CI_lower','CI_upper','CI_valid_draws']].to_string(index=False))
