"""Local gamma-model information with known or estimated gap exposures."""
from collections import defaultdict
from datetime import date
import io
from pathlib import Path
import subprocess
import sys
import zipfile

from lxml import etree
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]/"1.3"))
import table8_bootstrap as table

THETA = .5
GROUPS = ("two-dose", "unvaccinated", "all-persons")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def gamma_inverse(hazard, theta):
    return np.expm1(theta*hazard)/theta


def quiet_segments(mask):
    points = np.flatnonzero(mask)
    return np.split(points, np.where(np.diff(points)>1)[0]+1)


def read_czech():
    """Aggregate the actual CMR rows using the production decade/dose grouping."""
    path = ROOT/"data/Czech/KCOR_CMR.xlsx"
    with zipfile.ZipFile(path) as book:
        strings = ["".join(el.itertext()) for el in
                   etree.fromstring(book.read("xl/sharedStrings.xml")).findall(NS+"si")]
        workbook = etree.fromstring(book.read("xl/workbook.xml"))
        relationships = etree.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"):r.get("Target") for r in relationships}
        for enrollment,max_dose in (("2021_24",2),("2022_06",3)):
            sheet = next(s for s in workbook.findall(".//"+NS+"sheet") if s.get("name")==enrollment)
            target = targets[sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")]
            target = target.lstrip("/") if target.startswith("/") else "xl/"+target
            start_iso = enrollment.replace("_","-")
            groups = defaultdict(lambda: [0,0])
            with book.open(target) as stream:
                for _,row in etree.iterparse(stream,events=("end",),tag=NS+"row"):
                    if row.get("r")!="1":
                        week = strings[int(row[0].findtext(NS+"v"))]
                        if week>=start_iso:
                            cells = {c.get("r")[0]:c.findtext(NS+"v") for c in row}
                            yob,dose = int(cells["B"]),int(cells["E"])
                            if 1940<=yob<1970 and 0<=dose<=max_dose:
                                out = groups[yob//10*10,dose,week]
                                out[0] += int(cells["J"])
                                out[1] += int(cells["G"])
                    row.clear()
                    while row.getprevious() is not None:
                        del row.getparent()[0]
            datasets = defaultdict(dict)
            for (yob,dose,week),counts in groups.items():
                datasets[yob,dose][week] = counts
            for (yob,dose),counts in sorted(datasets.items()):
                dates = pd.DatetimeIndex([date.fromisocalendar(*map(int,w.split("-")),1)
                                          for w in sorted(counts)])
                values = np.array([counts[w] for w in sorted(counts)])
                yield "Czech",enrollment,str(yob),dose,dates,values[:,0],values[:,1]


def read_table8():
    workbook = subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=HERE)
    for sheet in table.SHEETS:
        dates,counts = table.read_population(io.BytesIO(workbook),sheet)
        counts = counts.copy()
        counts[2] += counts[0]+counts[1]
        alive = np.cumsum(counts[:,:,::-1],axis=2)[:,:,::-1][:,:,:-1]
        for group,dose in np.ndindex((3,3)):
            yield "Table8",sheet,GROUPS[group],dose,dates,alive[group,dose],counts[group,dose,:-1]


def calculate(kind,enrollment,group,dose,dates,risk,dead):
    model = table.K
    skip = model.DYNAMIC_HVE_SKIP_WEEKS
    time = np.arange(len(dates))
    quiet,_,_ = model._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(),time)
    risk,dead,quiet = risk[skip:],dead[skip:],quiet[skip:]
    segments = quiet_segments(quiet)
    theta = THETA
    gamma = table.CONFIG["gamma_per_week"]
    F = np.expm1(gamma*np.arange(len(risk)+1))/gamma
    h = -np.log1p(-np.clip(dead/(risk+model.EPS),0,.999))
    observed = np.r_[0.,np.cumsum(h)]

    # Choose a low-frailty reference rate using the quiet conditional hazards.
    # For the exact proportional gamma model, at a quiet-week opening the
    # mean surviving frailty is exp(-theta*H_obs). This supplies a reference
    # rate even if earlier gap exposures are unknown.
    deltaF = np.diff(F)
    def negative_log_likelihood(log_k):
        q = np.log1p(theta*np.exp(log_k)*deltaF[quiet]*np.exp(-theta*observed[:-1][quiet]))/theta
        p = -np.expm1(-q)
        return -np.sum(dead[quiet]*np.log(p)+(risk[quiet]-dead[quiet])*(-q))
    k = np.exp(minimize_scalar(negative_log_likelihood,bounds=(-20,-1),method="bounded").x)

    # Build the likelihood from quiet weeks plus one aggregated survival
    # transition across each gap. Each later quiet-window cumulative offset
    # is a nuisance parameter. Include the gap before the first quiet window
    # if modeled follow-up begins before that window.
    intervals = []
    offsets = []
    previous_end,previous_segment = 0,-1
    for segment,indices in enumerate(segments):
        start,end = int(indices[0]),int(indices[-1]+1)
        if start>previous_end:
            prev_x = k*F[previous_end]+(offsets[previous_segment] if previous_segment>=0 else 0.)
            prev_h = np.log1p(theta*prev_x)/theta
            start_x = gamma_inverse(prev_h+observed[start]-observed[previous_end],theta)
            offsets.append(start_x-k*F[start])
            intervals.append((previous_end,start,previous_segment,segment,False))
        else:
            offsets.append(0.)
        intervals.extend((i,i+1,segment,segment,True) for i in indices)
        previous_end,previous_segment = end,segment

    starts,ends,from_seg,to_seg,quiet_rows = map(np.array,zip(*intervals))
    quiet_rows = quiet_rows.astype(bool)
    n = risk[starts.astype(int)]
    offsets = np.asarray(offsets)
    free = np.flatnonzero(np.array([seg[0]>0 for seg in segments]))

    def integrated_hazard(parameters):
        rate,variance = np.exp(parameters[0]),parameters[1]
        delta = offsets.copy()
        delta[free] = parameters[2:]
        delta = np.r_[delta,0.]  # segment -1 is the modeled origin
        x = rate*F[starts.astype(int)]+delta[from_seg.astype(int)]
        y = rate*F[ends.astype(int)]+delta[to_seg.astype(int)]
        return np.log1p(variance*(y-x)/(1+variance*x))/variance

    parameters = np.r_[np.log(k),theta,offsets[free]]
    q = integrated_hazard(parameters)
    assert np.isfinite(q).all() and (q>0).all(), (kind,enrollment,group,dose)
    p = -np.expm1(-q)
    J = []
    for index in range(len(parameters)):
        step = 1e-5 if index<2 else 1e-6
        x = parameters.copy()
        x[index] += step
        y = parameters.copy()
        y[index] -= step
        J.append((integrated_hazard(x)-integrated_hazard(y))/(2*step))
    J = np.column_stack(J)
    weight = n*(1-p)/p

    def se(rows,columns):
        jac = J[np.ix_(np.flatnonzero(rows),columns)]*np.sqrt(weight[rows,None])
        signal = jac[:,columns.index(1)]
        nuisance = jac[:,[i for i,c in enumerate(columns) if c!=1]]
        if nuisance.shape[1]:
            nuisance = nuisance/np.linalg.norm(nuisance,axis=0)
            # Project the frailty derivative out of the nuisance span. This
            # also handles the rate/offset nonidentifiability in delayed
            # windows when gap observations are deliberately omitted.
            signal = signal-nuisance@np.linalg.lstsq(nuisance,signal,rcond=1e-8)[0]
        return float(1/np.linalg.norm(signal))

    first = quiet_rows & (from_seg==0)
    first_se = se(first,[0,1])
    all_known = se(quiet_rows,[0,1])
    known_rate_offsets = se(quiet_rows,[1])
    known_with_gaps = se(np.ones(len(q),bool),[0,1])
    unknown_quiet_only = se(quiet_rows,list(range(len(parameters))))
    unknown = se(np.ones(len(q),bool),list(range(len(parameters))))
    return dict(source=kind,enrollment=enrollment,group=group,dose=dose,people=int(risk[0]),
        first_weeks=len(segments[0]),quiet_weeks=int(quiet.sum()),gap_count=len(free),
        first_deaths=int(dead[segments[0]].sum()),quiet_deaths=int(dead[quiet].sum()),
        reference_rate=k,reference_theta=theta,first_se_known_offset=first_se,
        all_se_known_offsets=all_known,all_se_known_rate_and_offsets=known_rate_offsets,
        all_se_known_offsets_and_gap_counts=known_with_gaps,
        all_se_unknown_offsets_quiet_only=unknown_quiet_only,all_se_unknown_offsets=unknown,
        local_resolution_50=1.96*unknown/.5,local_resolution_25=1.96*unknown/.25)


def main():
    table.initialize([],"entry-moment")
    results = [calculate(*row) for row in read_table8()]
    results.extend(calculate(*row) for row in read_czech())
    frame = pd.DataFrame(results)
    frame.to_csv(HERE/"multiwindow-information.csv",index=False)
    print(frame[["source","enrollment","group","dose","quiet_weeks","first_se_known_offset",
                 "all_se_known_offsets","all_se_unknown_offsets","local_resolution_50"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
