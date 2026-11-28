from pathlib import Path
import os, shutil, json, hashlib, ast
ROOT=Path(__file__).resolve().parents[3]
a=Path(__file__).resolve().parent
dest=Path(os.environ["TEMP"])/"kcor-s4-approved"
current=dest/"m94"
def write(rel,data):
 p=current/rel;p.parent.mkdir(parents=True,exist_ok=True);p.unlink(missing_ok=True);p.write_bytes(data if isinstance(data,bytes) else data.encode("utf8"))
revised=ROOT/"iterative review/revised"
for e in json.loads((revised/"manifest.json").read_text()):
 p=revised/e["path"];b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==e["sha256"];write(e["path"],b)
p=current/"code/KCOR.py";s=p.read_text(encoding="utf-8-sig")
def fn(src,name):
 n=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.get_source_segment(src,n)
base=fn(s,"fit_theta0_gompertz")
exp=(ROOT/"objections/2.7/low-frailty/all-window-joint/experiment.py").read_text(encoding="utf-8-sig")
node=next(n for n in ast.walk(ast.parse(exp)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "def _residual_joint(params)" in n.value)
a0=base.index("        # 3c:");b0=base.index("        except Exception as e:",a0)
new=base[:a0]+node.value+base[b0:]
new=new.replace("def fit_theta0_gompertz(","def _fit_all_window_gamma(").replace("joint_seed=False,","joint_seed=True,\n    account_entry=False,")
new=new.replace("if PROFILE:","if True:").replace("if MULTISTART:","if True:").replace("theta_new = float(r.x[0])","theta_new = float(r.x[1])")
new=new.replace("if abs(theta_new - theta) < convergence_tol:","if r.success or (abs(theta_new - theta) < convergence_tol and abs(np.log(k_fixed/old_k)) < convergence_tol):")
seed="            H_gom_full = k_fixed * unit_start"; assert new.count(seed)==1
new=new.replace(seed,"""            if account_entry and first_seg_start > post_idx[0]:
                before = post_idx[post_idx < first_seg_start]
                observed = float(np.sum(h_arr[before]))
                log_ratio = theta_seed * observed - np.log1p(theta_seed * k_fixed * unit_start[first_seg_start])
                k_fixed = np.exp(min(np.log(0.1), np.log(k_fixed) + max(0., log_ratio)))
"""+seed)
bound="    reentry_indices = [start for start, _ in segments[1:]]";assert new.count(bound)==1
new=new.replace(bound,bound+"\n    if account_entry and joint_seed and first_seg_start > post_idx[0]:\n        reentry_indices.insert(0, first_seg_start)")
dstart=new.index('    """');dend=new.index('    """',dstart+7)+7
new=new[:dstart]+'''    """Joint common-k/theta fit over all quiet windows with profiled gap offsets.
    The first-window joint fit initializes the optimization, not a fixed final k.
    account_entry includes proportional exposure before the first quiet window.
    """'''+new[dend:]
s=s.replace(base,new)
trial=(ROOT/"objections/1.2/weekly-survival/trial.py").read_text(encoding="utf-8-sig")
fit=(ROOT/"objections/3.2/nonlinear-survivor-fit/fitter.py").read_text(encoding="utf-8-sig")
body=fit[fit.index("# Numerically stable, normalized gamma quadrature."):]
body=body.replace('base=candidate(model,"multistart")','base=model._fit_all_window_gamma')
body=body.replace("reference_hazard(","cumulative_hazard(").replace('message="nonlinear survivor trial"','message="nonlinear survivor fit"')
module=trial.replace("Experimental weekly inversion","Weekly inversion").replace('correction_mode="weekly_survival_trial"','correction_mode="weekly_survival"')+"\n\nfrom scipy.linalg import eigh_tridiagonal\nfrom scipy.special import gammaln\nfrom scipy.optimize import least_squares\nimport math\n\n"+body
write("code/kcor_survivor.py",module)
s=s.replace("def apply_nph_correction_post_inversion(","def _apply_nph_identity_or_fallback(",1)
wrappers="""

# Shared all-window and nonlinear survivor model.
def _survivor_module():
    import sys
    from pathlib import Path
    parent = str(Path(__file__).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import kcor_survivor
    return kcor_survivor


def fit_theta0_gompertz(h_arr, t_rebased, quiet_mask, k_anchor_weeks,
                       gamma_per_week, theta0_init=1.0, deaths_arr=None, *,
                       joint_seed=True, iso_week=None, wave=None, alpha=None):
    \"\"\"Fit baseline and frailty jointly; use the specified nonlinear wave if active.\"\"\"
    from types import SimpleNamespace
    t = np.asarray(t_rebased)
    if wave is None and iso_week is not None:
        cfg = get_covid_correction_config()
        if cfg and cfg["apply_correction"]:
            a = cfg["start_dt"].isocalendar()
            b = cfg["end_dt"].isocalendar()
            iso = np.asarray(iso_week)
            active = (iso >= a.year*100+a.week) & (iso <= b.year*100+b.week) & (t >= 0)
            if active.any():
                wave = (float(t[active][0]), float(t[active][-1]))
                alpha = cfg["apply_alpha"]
    if wave is not None and alpha is not None and abs(alpha-1.) > NPH_IDENTITY_TOL:
        model = SimpleNamespace(**globals())
        fit = _survivor_module().make_fitter(model, wave, float(alpha))
        return fit(h_arr, t_rebased, quiet_mask, k_anchor_weeks, gamma_per_week,
                   theta0_init, deaths_arr)
    return _fit_all_window_gamma(h_arr, t_rebased, quiet_mask, k_anchor_weeks,
                                 gamma_per_week, theta0_init, deaths_arr,
                                 joint_seed=True, account_entry=True)


def apply_nph_correction_post_inversion(*args, **kwargs):
    from types import SimpleNamespace
    model = SimpleNamespace(**globals())
    model.apply_nph_correction_post_inversion = _apply_nph_identity_or_fallback
    return _survivor_module().make_correction(model)(*args, **kwargs)


"""
tree=ast.parse(s);lines=s.splitlines(keepends=True)
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="fit_theta0_gompertz"]
assert len(calls)==3,len(calls)
for n in sorted(calls,key=lambda q:q.end_lineno,reverse=True):
 idx=n.end_lineno-1
 indent=lines[idx][:len(lines[idx])-len(lines[idx].lstrip())]+"    "
 lines.insert(idx,indent+"iso_week=iso_int,\n")
s="".join(lines)
s=s.replace('if __name__ == "__main__":',wrappers+'if __name__ == "__main__":',1)
ast.parse(s);ast.parse(module);write("code/KCOR.py",s)
print("Integrated fitter, proportional entry and exact nonlinear wave routing.",flush=True)
(a/"candidate-location.txt").write_text(str(dest),encoding="utf8")