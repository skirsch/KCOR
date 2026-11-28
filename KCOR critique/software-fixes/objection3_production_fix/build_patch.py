"""Build a separate submission tree and patch; never edit repository source."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED = '3d9e0320739baa3d11b1058e8b134335f7edcd9a0d7df6ec22234c2abf1f49ef'


def replace_once(text, old, new):
    assert text.count(old) == 1, f'Expected one match: {old[:90]}'
    return text.replace(old, new, 1)


def main():
    raw = (ROOT/'code/KCOR.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Production source changed; review patch base first'
    old = raw.decode('utf-8').replace('\r\n', '\n')
    new = old
    replacements = [
        ('        df["t"]    = df.groupby(["YearOfBirth","Dose"]).cumcount().astype(float)',
         '        time_group_cols = ["YearOfBirth", "Dose"]\n'
         '        if MONTE_CARLO_MODE and "mc_id" in df.columns:\n'
         '            time_group_cols = ["mc_id"] + time_group_cols\n'
         '        df["t"] = df.groupby(time_group_cols).cumcount().astype(float)'),
        ('                for dose in unique_doses:\n                    # Determine age_group label',
         '                mc_ids = [None]\n'
         '                if MONTE_CARLO_MODE and "mc_id" in yob_data.columns:\n'
         '                    mc_ids = sorted(yob_data["mc_id"].dropna().unique().tolist())\n'
         '                for mc_id, dose in ((i, d) for i in mc_ids for d in unique_doses):\n'
         '                    # Determine age_group label'),
        ('                    params_key = (enrollment_date, int(year_of_birth), int(dose))\n                    params = kcor6_params_map.get(params_key)',
         '                    params_key = (enrollment_date, int(year_of_birth), int(dose))\n'
         '                    if mc_id is not None:\n'
         '                        params_key = (enrollment_date, int(mc_id), int(year_of_birth), int(dose))\n'
         '                    params = kcor6_params_map.get(params_key)'),
        ('                        # Only include if we have valid fit parameters\n                        if np.isfinite(theta0_hat) or np.isfinite(k_hat):\n                            gamma_frailty_rows.append({',
         '                        # Preserve failed Monte Carlo fits as diagnostic evidence.\n'
         '                        if mc_id is not None or np.isfinite(theta0_hat) or np.isfinite(k_hat):\n'
         '                            gamma_frailty_rows.append({\n'
         '                                **({"mc_id": int(mc_id), "theta0_status": params.get("theta0_status", "")} if mc_id is not None else {}),'),
        ('            x["year_of_birth"],\n            x["dose"]\n        ))',
         '            x["year_of_birth"],\n            x["dose"],\n            x.get("mc_id", -1)\n        ))'),
        ('    def _theta0_params_lookup(enrollment_date, yob, dose):\n        if not isinstance(kcor6_params_map, dict):\n            return None\n        params = kcor6_params_map.get((enrollment_date, int(yob), int(dose)))',
         '    def _theta0_params_lookup(enrollment_date, yob, dose, mc_id=None):\n'
         '        if not isinstance(kcor6_params_map, dict):\n            return None\n'
         '        key = (enrollment_date, int(yob), int(dose))\n'
         '        if MONTE_CARLO_MODE and mc_id is not None and pd.notna(mc_id):\n'
         '            key = (enrollment_date, int(mc_id), int(yob), int(dose))\n'
         '        params = kcor6_params_map.get(key)'),
        ('                            if not isinstance(key, tuple) or len(key) != 3:\n                                continue\n                            if str(key[0]) != str(sheet_name):\n                                continue\n                            st = _theta0_status_from_params(params)',
         '                            if not isinstance(key, tuple) or len(key) not in (3, 4):\n'
         '                                continue\n'
         '                            if str(key[0]) != str(sheet_name):\n'
         '                                continue\n'
         '                            mc_id = key[1] if len(key) == 4 else None\n'
         '                            st = _theta0_status_from_params(params)'),
        ('                                    yob_val = int(key[1])\n                                    dose_val = int(key[2])',
         '                                    yob_val = int(key[-2])\n                                    dose_val = int(key[-1])'),
        ('                                    quality_hits.add((yob_val, dose_val, st))',
         '                                    quality_hits.add((mc_id, yob_val, dose_val, st))'),
        ('                    quality_hits_sorted = sorted(quality_hits)',
         '                    quality_hits_sorted = sorted(quality_hits, key=lambda item: (-1 if item[0] is None else item[0], *item[1:]))'),
        ('                        f"({int(yob_val)},{int(dose_val)},{str(status_val)})"\n                        for yob_val, dose_val, status_val in quality_hits_sorted',
         '                        (f"(mc_id={int(mc_id)},{int(yob_val)},{int(dose_val)},{str(status_val)})"\n'
         '                         if mc_id is not None else f"({int(yob_val)},{int(dose_val)},{str(status_val)})")\n'
         '                        for mc_id, yob_val, dose_val, status_val in quality_hits_sorted'),
        ('                            params_num = _theta0_params_lookup(sheet_name, age, dose_num)\n                            params_den = _theta0_params_lookup(sheet_name, age, dose_den)',
         '                            mc_id = row.get("mc_id") if MONTE_CARLO_MODE else None\n'
         '                            params_num = _theta0_params_lookup(sheet_name, age, dose_num, mc_id)\n'
         '                            params_den = _theta0_params_lookup(sheet_name, age, dose_den, mc_id)'),
        ('                                "Row_Theta0_NonOK": row_theta_non_ok,',
         '                                "Row_Theta0_NonOK": row_theta_non_ok,\n'
         '                                **({"mc_id": int(mc_id)} if mc_id is not None and pd.notna(mc_id) else {}),'),
        ('                        "reporting_date", "enrollment_date", "age_group", "year_of_birth", "dose",',
         '                        "reporting_date", "enrollment_date", "age_group", "year_of_birth", "dose", "mc_id", "theta0_status",'),
    ]
    for before, after in replacements:
        new = replace_once(new, before, after)
    tree = HERE/'submission'
    (tree/'code').mkdir(parents=True, exist_ok=False)
    (tree/'data/Czech').mkdir(parents=True, exist_ok=False)
    newline = '\r\n' if b'\r\n' in raw else '\n'
    patched = new.replace('\n', newline).encode('utf-8')
    (tree/'code/KCOR.py').write_bytes(patched)
    (tree/'data/Czech/Czech.yaml').write_bytes((ROOT/'data/Czech/Czech.yaml').read_bytes())
    test_rel = 'test/monte_carlo/test_sample_isolation.py'
    test = (HERE/'test_sample_isolation.py').read_text(encoding='utf-8')
    (tree/test_rel).parent.mkdir(parents=True, exist_ok=False)
    (tree/test_rel).write_text(test, encoding='utf-8')
    diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                      fromfile='a/code/KCOR.py', tofile='b/code/KCOR.py'))
    diff += ''.join(difflib.unified_diff([], test.splitlines(True), fromfile='/dev/null', tofile='b/'+test_rel))
    with (HERE/'KCOR_MC_sample_isolation.patch').open('x', encoding='utf-8', newline='\n') as f:
        f.write(diff)
    manifest = dict(original_sha256=EXPECTED, patched_sha256=hashlib.sha256(patched).hexdigest(),
                    test_sha256=hashlib.sha256(test.encode()).hexdigest(), original_modified=False,
                    patch_files=['code/KCOR.py', test_rel])
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Built isolated patch and submission tree; original source unchanged.')


if __name__ == '__main__':
    main()
