"""Plot the observed historical pooled objective sensitivity; no fabricated data."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
df=pd.read_csv(p/"wider-grid-curves.csv")
df=df[df.scenario=="all"]
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,2,figsize=(11,4.7),layout="constrained",sharex=True)
colors={"historical":"#666666","working":"#0072B2","exact":"#D55E00"}
labels={"historical":"Historical gamma proxy","working":"All-window fit + gamma proxy","exact":"Alpha-dependent fit + exact moments"}
for panel,name in zip(ax,("pairwise","collapse")):
    panel.axvspan(.7,1.,color="#f1f1f1",zorder=0)
    for arm in labels:
        g=df[(df.arm==arm)&(df.estimator==name)].sort_values("alpha")
        y=g.objective-g.objective.min()
        panel.plot(g.alpha,y,color=colors[arm],lw=2,label=labels[arm])
        panel.scatter(g.alpha.iloc[np.argmin(y)],0,color=colors[arm],s=35,zorder=3)
    panel.axvline(1.,color="#222222",lw=.8,ls="--")
    panel.set_title(name.capitalize()+" objective");panel.set_xlabel("Nonlinear exponent alpha")
    panel.set_ylabel("Objective above its minimum")
    panel.grid(alpha=.18);panel.set_xlim(.695,1.305)
ax[0].legend(loc="best",fontsize=9,frameon=False)
fig.suptitle("Pooled Czech alpha objective: original and experimental fits",fontsize=15,fontweight="bold")
fig.text(.5,.005,"Gray shading: added lower-grid sensitivity (outside the historical 1.00–1.30 grid).",ha="center",fontsize=9)
fig.savefig(p/"pooled-alpha-objectives.png",dpi=180,bbox_inches="tight")
