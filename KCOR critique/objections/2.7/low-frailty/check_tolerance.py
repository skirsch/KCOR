"""Rescale the least-squares residual by a constant to isolate stopping tolerance."""
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd

import investigate as inv


def compare(task):
    size, rep = task
    theta = .5
    rng = np.random.default_rng([20261127, inv.SIZES.index(size), inv.FRAILTIES.index(theta), rep])
    counts = rng.multinomial(size, inv.probabilities(theta))
    alive = np.cumsum(counts[::-1])[::-1][:-1]
    h = -np.log1p(-counts[:-1]/alive)
    model = inv.study.model
    original_solver = model.least_squares
    rows = []

    def scaled(residual, *args, **kwargs):
        return original_solver(lambda parameters: residual(parameters)/inv.RATE, *args, **kwargs)

    for length in (4,52):
        for label,fit in (("fixed",inv.study.compare.production_fit),("joint",inv.study.compare.joint_fit)):
            args = (h,inv.TIME,inv.TIME<length,4,inv.GAMMA)
            _, old = fit(*args,deaths_arr=counts[:-1])
            with patch.object(model,"least_squares",scaled):
                _, new = fit(*args,deaths_arr=counts[:-1])
            rows.append(dict(size=size,rep=rep,weeks=length,fitter=label,
                             old_theta=old["theta0_raw"],scaled_theta=new["theta0_raw"],
                             old_status=old["status"],scaled_status=new["status"]))
    return rows


if __name__ == "__main__":
    tasks = [(size,rep) for size in (50_000,5_000_000) for rep in range(200)]
    rows = []
    with ProcessPoolExecutor(7,initializer=inv.initialize) as pool:
        for result in pool.map(compare,tasks,chunksize=10):
            rows.extend(result)
    df = pd.DataFrame(rows)
    saved = pd.read_csv(inv.HERE/"fit-estimates.csv")
    keys = ["size","rep","weeks","fitter"]
    matched = df.merge(saved[saved.theta_true==.5][keys+["theta_raw"]],on=keys,validate="one_to_one")
    np.testing.assert_allclose(matched.old_theta,matched.theta_raw,atol=1e-10)
    df.to_csv(inv.HERE/"tolerance-comparison.csv",index=False)
    df["abs_difference"] = abs(df.old_theta-df.scaled_theta)
    print(df.groupby(["size","weeks","fitter"])[["old_theta","scaled_theta","abs_difference"]].agg(["mean","max"]).to_string())
