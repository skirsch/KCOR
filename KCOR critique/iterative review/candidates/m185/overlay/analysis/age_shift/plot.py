"""Render the two age-shift negative controls from the matched bootstrap replay."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    f=pd.read_csv(HERE/"curves.csv")
    for pair,title,name in [
        ("1/0","10-year age shift (1940-49 vs 1930-39)","fig2_neg_control_10yr_age_diff.png"),
        ("2/0","20-year age shift (1950-59 vs 1930-39)","fig3_neg_control_20yr_age_diff.png"),
    ]:
        g=f[f.pair==pair]
        fig,ax=plt.subplots(figsize=(7,3.8),layout="constrained")
        x=g.week_since_enrollment.to_numpy()
        ax.axhline(1,color="0.3",ls="--",lw=1,label="Unity")
        ax.fill_between(x,g.lower.to_numpy(),g.upper.to_numpy(),color="#2879a8",alpha=.23,label="Pointwise 95% interval")
        ax.plot(x,g.point.to_numpy(),color="#21618a",lw=1.8,label="Anchored KCOR")
        ax.set(xlabel="Weeks since enrollment",ylabel="Anchored KCOR",title=title,xlim=(5,173),ylim=(.8,1.2))
        ax.grid(axis="y",alpha=.23)
        ax.legend(loc="upper right",fontsize=8,frameon=True)
        fig.savefig(ROOT/"documentation/preprint/figures"/name,dpi=250)
        plt.close(fig)
if __name__=="__main__":main()
