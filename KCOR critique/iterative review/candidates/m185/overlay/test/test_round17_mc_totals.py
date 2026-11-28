"""Round 17.1: direct all-age MC populations are selected independently."""
from test_round16_mc_pooling import K, S, input_frame, run, pd, np
import os


def supplied_totals(f, doses=None):
    components = f if doses is None else f[f.Dose.isin(doses)]
    return components.groupby(['mc_id','Dose','DateDied'], as_index=False).agg(
        ISOweekDied=('ISOweekDied','first'), Alive=('Alive','sum'), Dead=('Dead','sum')
    ).assign(YearOfBirth=-2)


def chosen(out, draw):
    return out[out.mc_id.eq(draw)&out.YearOfBirth.isin([0,-2])].sort_values(['YearOfBirth','Date']).reset_index(drop=True)


def main():
    f=input_frame(S); ix=f[f.YearOfBirth.eq(1940)&f.Dose.eq(1)].index
    f.loc[ix[8:],'Dead']*=3; f.loc[ix,'Alive']=20000-np.r_[0,np.cumsum(f.loc[ix,'Dead'].to_numpy()[:-1])]
    a=f.assign(mc_id=1)
    b=input_frame(S).assign(mc_id=2)
    b.loc[b.YearOfBirth.eq(1950),['Alive','Dead']]*=15
    # Same replicate B has both exact totals, or only one supplied total.
    # A remains component-only in all configurations.
    b_full=pd.concat([b,supplied_totals(b)],ignore_index=True)
    b_one=pd.concat([b,supplied_totals(b,[0])],ignore_index=True)
    reference_a,_=run(a)
    reference_b,_=run(b)
    expected_a=chosen(reference_a,1)
    assert set(expected_a.YearOfBirth)=={0,-2} and len(expected_a)==40
    for label,part in [('full',b_full),('one-dose',b_one)]:
        reference,_=run(part)
        expected_b=chosen(reference,2)
        assert set(expected_b.YearOfBirth)=={0,-2} and len(expected_b)==40
        np.testing.assert_allclose(chosen(reference_b,2).KCOR,expected_b.KCOR,rtol=1e-13,equal_nan=True)
        mixed=pd.concat([a,part],ignore_index=True)
        batch,_=run(mixed)
        shuffled,_=run(mixed.sample(frac=1,random_state=99).reset_index(drop=True))
        for expected,i in [(expected_a,1),(expected_b,2)]:
            for out in (batch,shuffled):
                pd.testing.assert_frame_equal(expected,chosen(out,i),check_exact=True)
        written,_=run(mixed,write=True)
        for expected,i in [(expected_a,1),(expected_b,2)]:
            actual=chosen(written,i)
            np.testing.assert_allclose(expected.KCOR,actual.KCOR,rtol=1e-14,equal_nan=True)
            np.testing.assert_allclose(expected.KCOR_unanchored,actual.KCOR_unanchored,rtol=1e-14,equal_nan=True)
            assert len(actual)==40
        print('PASS',label,'standalone/batched/shuffled exact parity, both all-ages and ASMR; written workbook parity')
    # Partial scope preserves the existing refusal to mix a wider supplied
    # total with a reconstructed incomplete component population.
    os.environ['KCOR_AGGREGATE_SCOPE']='partial'
    combined=pd.concat([a,b_one],ignore_index=True)
    selected=combined[K._all_age_source_mask(combined)]
    assert not selected[selected.mc_id.eq(2)].Dose.eq(1).any()
    assert len(selected[selected.mc_id.eq(1)])==len(a)
    print('PASS per-replicate partial-scope source rule')


if __name__=='__main__': main()
