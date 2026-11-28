import pandas as pd
from pathlib import Path
p=Path(__file__).resolve().parent
r=pd.read_csv(p/"runs.csv")
z=r[(r.rep>=0)&(r.epsilon_comorb==0)]
print(z.groupby(["size","fitter"]).KCOR_asymptote.agg(["mean","std",lambda x:x.quantile(.025),lambda x:x.quantile(.975)]).round(5).to_string())
print("flags/failures",r[["success_a","success_b","flag_a","flag_b"]].sum())
