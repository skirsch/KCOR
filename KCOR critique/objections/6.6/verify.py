"""Verify interval risk sets passed to the existing Cox library."""
import argparse,ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
from unittest.mock import patch
import numpy as np
import pandas as pd
import statsmodels.api

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate",type=Path);args=parser.parse_args()
    p=args.candidate/"test/sim_grid/code/generate_sim_grid.py"
    tree=ast.parse(p.read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="fit_time_varying_cox")
    ns=dict(np=np,pd=pd,Dict=Dict,SimConfig=object)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),"exec"),ns)
    captured=[]
    class CapturePHReg:
        def __init__(self,time,exog,status):
            captured.append(dict(count=len(time),events=int(sum(status)),max_time=float(max(time))))
        def fit(self,disp=False):return SimpleNamespace(params=np.array([0.]))
    for fractional in (False,True):
        captured.clear()
        dead=np.zeros(100,float)
        if fractional:
            dead[:20]=.6
            dead[20:50]=.6
        else:
            dead[0]=10;dead[25]=10
        alive=100-np.r_[0,np.cumsum(dead[:-1])]
        cohorts=[dict(dose=d,weeks=np.arange(100),alive=alive,dead=dead) for d in (0,1)]
        with patch("statsmodels.duration.hazard_regression.PHReg",CapturePHReg):
            result=ns["fit_time_varying_cox"](dict(cohorts=cohorts),SimpleNamespace(n_weeks=100))
        expected=[(200,24,20),(176,36,30)] if fractional else [(200,20,20),(180,20,30)]
        assert [(x["count"],x["events"],x["max_time"]) for x in captured]==expected,captured
        assert result["cox_fit_success"]
        print("fractional",fractional,captured)
if __name__=="__main__":main()
