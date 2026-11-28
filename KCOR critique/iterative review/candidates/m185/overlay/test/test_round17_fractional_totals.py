"""Exhaustive total sums: fractional roundoff only, exact integer checks."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'test'),str(ROOT/'code')]
from test_round15_nested_totals import input_frame,total,ending,written,S
from fixed_cohort_history import aggregate_reconciliation,invalid_components


def main():
    f=input_frame(S);f.loc[f.YearOfBirth.eq(1950),'YearOfBirth']=1960
    for age,opening in [(1940,20001),(1960,10003)]:
        for dose in (0,1):
            ix=f.index[f.YearOfBirth.eq(age)&f.Dose.eq(dose)]
            f.loc[ix,'Alive']=opening-np.r_[0,np.cumsum(f.loc[ix,'Dead'].to_numpy()[:-1])]
    full=total(f);young=total(f[f.YearOfBirth.eq(1960)]).assign(YearOfBirth=-60)
    supplied=pd.concat([f,full,young],ignore_index=True)
    assert aggregate_reconciliation(supplied)==({},set())
    wrong=supplied.copy();wrong.loc[wrong.YearOfBirth.eq(-2)&wrong.Dose.eq(1),'Alive']+=1
    assert set(aggregate_reconciliation(wrong)[0])=={(-2,1)}
    fractional=supplied.copy();fractional[['Alive','Dead']]=fractional[['Alive','Dead']].astype(float)/10
    comp=fractional[fractional.YearOfBirth.gt(0)]
    assert not invalid_components(fractional,fractional.DateDied.min())
    x=fractional[fractional.YearOfBirth.eq(-2)].groupby(['Dose','DateDied'])[['Alive','Dead']].sum()
    y=comp.groupby(['Dose','DateDied'])[['Alive','Dead']].sum()
    assert not np.array_equal(x.to_numpy(),y.to_numpy())
    assert aggregate_reconciliation(fractional)==({},set())
    assert aggregate_reconciliation(fractional,'partial')==({},set())
    wrong=fractional.copy();wrong.loc[wrong.YearOfBirth.eq(-2)&wrong.Dose.eq(1),'Alive']+=.001
    assert set(aggregate_reconciliation(wrong)[0])=={(-2,1)}
    # Actual written workbook, point-only for fractional expected counts.
    baseline=written(comp,'exhaustive',0)
    accepted=written(fractional,'exhaustive',0)
    rejected=written(wrong,'exhaustive',0)
    for age in (1940,1960,-2):
        np.testing.assert_allclose(ending(baseline,age).KCOR,ending(accepted,age).KCOR,rtol=1e-12)
    assert rejected[rejected.YearOfBirth.eq(-2)].KCOR.isna().all()
    assert rejected[rejected.YearOfBirth.eq(-2)].normalization_status.eq('invalid_aggregate_counts').all()
    for age in (1940,1960,-60):
        np.testing.assert_allclose(ending(accepted,age).KCOR,ending(rejected,age).KCOR,rtol=1e-12)
    print('Max valid sum difference:',np.max(np.abs(x.to_numpy()-y.to_numpy())))
    print('Point-only full-age ratios (reconstructed, supplied):',ending(baseline,-2).KCOR,ending(accepted,-2).KCOR)
    print('Material fractional mismatch and one-person integer mismatch rejected; unrelated ages retained.')

if __name__=='__main__':main()
