"""Read-only observation of completed age-specific production output.

Execute in the live production verifier with sys.remote_exec(pid, this_path).
The observation uses an isolated namespace and changes no numerical inputs or
production variables. Capturing is allowed only after all 1,000 age curves exist.
"""
exec(r'''
import sys, json, hashlib
from pathlib import Path
import pandas as pd
base = Path(__file__).resolve().parent / 'reproduction_1000_iterations'
source = base.parents[4] / 'code/KCOR.py'
matches = []
for current in sys._current_frames().values():
    while current is not None:
        if current.f_code.co_name == 'build_kcor_rows' and Path(current.f_code.co_filename).resolve() == source.resolve():
            matches.append(current)
        current = current.f_back
assert len(matches) == 1, 'Expected one live production build_kcor_rows frame'
production_frame = matches[0]
completed = production_frame.f_locals['out_rows']
assert len(completed) == 1000, 'Age-specific production calculations are not complete'
rows = pd.concat(completed, ignore_index=True).copy(deep=True)
assert len(rows) == 100000
assert rows.mc_id.nunique() == 1000
assert rows.YearOfBirth.eq(1950).all()
assert rows.Dose_num.eq(1).all() and rows.Dose_den.eq(0).all()
assert not rows[['mc_id', 'Date', 'YearOfBirth', 'Dose_num', 'Dose_den']].duplicated().any()
with (base/'production_age_rows.csv').open('x', encoding='utf-8', newline='') as stream:
    rows.to_csv(stream, index=False)
metadata = {
    'capture': 'completed age-specific out_rows from unmodified production build_kcor_rows',
    'rows': len(rows), 'iterations': 1000, 'weeks_per_iteration': 100,
    'production_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'paused_at_line': production_frame.f_lineno,
    'ongoing_work': 'all_ages_asmr_rows; source states these rows are not included in output',
    'csv_sha256': hashlib.sha256((base/'production_age_rows.csv').read_bytes()).hexdigest(),
    'numerical_inputs_modified': False,
}
with (base/'production_capture.json').open('x', encoding='utf-8') as stream:
    json.dump(metadata, stream, indent=2)
''', {})
