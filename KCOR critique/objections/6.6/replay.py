"""Replay only the historical Cox comparator on its six original scenarios."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent

def worker(task):
    candidate,index=task
    sys.path.insert(0,str(candidate/"test/sim_grid/code"))
    import generate_sim_grid as sim
    config=sim.SimConfig(seed=42)
    _,generate=sim._SIM_GRID_SCENARIO_SPECS[index]
    data=generate(config)
    # Verify the generated counts agree with the historical saved input.
    expected=sim.generate_kcor_cmr_format(data,config)
    workbook=candidate/"test/sim_grid/data/sim_grid_data.xlsx"
    existing=pd.read_excel(workbook,sheet_name=data["scenario"])
    for col in ("Alive","Dead","Dose"):
        np.testing.assert_allclose(expected[col],existing[col],rtol=1e-10,atol=1e-8)
    result=sim.fit_time_varying_cox(data,config)
    return dict(scenario=data["scenario"],**result)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("candidate",type=Path)
    args=p.parse_args()
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(worker,[(args.candidate,i) for i in range(6)]))
    for row in results:print(row,flush=True)
    (HERE/"cox-replay.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
if __name__=="__main__":main()
