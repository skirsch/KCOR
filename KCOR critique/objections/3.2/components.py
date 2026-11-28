"""Inspect baseline and attributed wave exposure over the five modeled 2022 wave weeks."""
import csv
import io
import subprocess
from concurrent.futures import ProcessPoolExecutor
from experiment import HERE, entry_fit, np, table


def initialize(populations):
    global FITS, ORIGINAL
    table.initialize(populations, "gap-refit")
    FITS = (table.compare.joint_fit, entry_fit(table.K))
    ORIGINAL = table.K.apply_nph_correction_post_inversion


def collect(draw):
    values = []
    for fit in FITS:
        rows=[]

        def capture(curve,t,dates,theta,alpha,start,end,k,gamma,**kwargs):
            rebased = t-table.K.DYNAMIC_HVE_SKIP_WEEKS
            use = (dates>=np.datetime64(start)) & (dates<=np.datetime64(end)) & (rebased>=0)
            assert use.sum()==5 and np.array_equal(rebased[use],np.arange(5))
            result,diag = ORIGINAL(curve,t,dates,theta,alpha,start,end,k,gamma,**kwargs)
            baseline = (k*np.exp(gamma*rebased[use])).sum()
            rows.append([baseline,diag["post_wave_exposure"]])
            return result,diag

        table.compare.joint_fit=fit
        table.K.apply_nph_correction_post_inversion=capture
        ratios,_,_=table.estimate(1,draw)
        assert len(rows)==9
        values.append((rows,ratios))
    return draw,values


if __name__ == "__main__":
    data=subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=HERE)
    populations=[table.read_population(io.BytesIO(data),s) for s in table.SHEETS]
    values=np.empty((2,1000,9,2))
    ratios=np.empty((2,1000,3))
    with ProcessPoolExecutor(4,initializer=initialize,initargs=(populations,)) as pool:
        for d,results in pool.map(collect,range(1000),chunksize=5):
            for arm,(rows,points) in enumerate(results):
                values[arm,d]=rows
                ratios[arm,d]=points
    np.testing.assert_allclose(ratios[0],np.load(HERE.parent/"3.1/table8-draws.npz")["points"][1,1],rtol=1e-12)
    np.testing.assert_allclose(ratios[1],np.load(HERE/"entry-draws.npz")["proposed"][1],rtol=1e-12)
    with (HERE/"components-2022.csv").open("w",newline="") as out:
        writer=csv.writer(out)
        writer.writerow(("subgroup","pseudo_dose","quantity","fit","point","lower95","upper95"))
        for g in range(9):
            for j,quantity in enumerate(("baseline","wave")):
                for arm,label in enumerate(("current","experimental")):
                    samples=values[arm,:,g,j]
                    writer.writerow((("two-dose","unvaccinated","all-persons")[g//3],g%3,quantity,
                                     label,samples[0],*np.quantile(samples[1:],[.025,.975])))
