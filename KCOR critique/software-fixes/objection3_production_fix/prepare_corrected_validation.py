"""Create an isolated corrected-code calibration without altering prior evidence."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT/'KCOR critique/objections/1.3/objection3_resolution'
NEW = ROOT/'KCOR critique/objections/1.3/objection3_corrected_validation'
PR = Path('C:/Users/jasmi/repo/KCOR-mc-fix-pr')
NEW.mkdir(exist_ok=False)
(NEW/'development').mkdir()
copied = {}
for name in ['production_adapter.py', 'simulation.py', 'run_study.py', 'summarize.py',
             'verify_production.py', 'check_model.py']:
    raw = (OLD/name).read_bytes()
    copied[name] = hashlib.sha256(raw).hexdigest()
    (NEW/name).write_bytes(raw)
for name in ['code/KCOR.py', 'code/KCOR_CMR.py', 'data/Czech/Czech.yaml']:
    dest = NEW/'source_snapshot'/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PR/name, dest)

path = NEW/'production_adapter.py'
s = path.read_text(encoding='utf-8')
s = s.replace('Isolated adapter to the unmodified KCOR v7 production functions.',
              'Isolated adapter to the corrected KCOR production functions from PR #4.')
s = s.replace("sys.path.insert(0, str(ROOT / 'code'))",
              "SOURCE_ROOT = Path(__file__).resolve().parent / 'source_snapshot'\nsys.path.insert(0, str(SOURCE_ROOT / 'code'))")
start = s.index('    # Current process_workbook groups cumcount')
end = s.index('    quiet =', start)
s = s[:start] + ('    # PR #4 groups time by mc_id, age, and dose. Each complete-grid\n'
    '    # sample therefore uses calendar weeks independent of batch size or ID.\n'
    '    t = calendar_t\n') + s[end:]
path.write_text(s, encoding='utf-8')

path = NEW/'run_study.py'
s = path.read_text(encoding='utf-8').replace('from production_adapter import ROOT,',
                                           'from production_adapter import ROOT, SOURCE_ROOT,')
s = s.replace("paths = [ROOT/'code/KCOR.py', ROOT/'code/KCOR_CMR.py', ROOT/'data/Czech/Czech.yaml']",
              "paths = [SOURCE_ROOT/'code/KCOR.py', SOURCE_ROOT/'code/KCOR_CMR.py', SOURCE_ROOT/'data/Czech/Czech.yaml']")
s = s.replace('Output must stay inside objection3_resolution', 'Output must stay inside objection3_corrected_validation')
path.write_text(s, encoding='utf-8')

path = NEW/'verify_production.py'
s = path.read_text(encoding='utf-8').replace('K, ROOT, START,', 'K, ROOT, SOURCE_ROOT, START,')
s = s.replace("ROOT/'code/KCOR.py', ROOT/'code/KCOR_CMR.py', ROOT/'data/Czech/Czech.yaml'",
              "SOURCE_ROOT/'code/KCOR.py', SOURCE_ROOT/'code/KCOR_CMR.py', SOURCE_ROOT/'data/Czech/Czech.yaml'")
s = s.replace('weeks=80), KCOR=v', "weeks=scenario['target']), KCOR=v")
s = s.replace('estimate(bootstrap(original, rng))', "estimate(bootstrap(original, rng), windows, scenario['target'])")
path.write_text(s, encoding='utf-8')

design = json.loads((OLD/'design.json').read_text())
design.update(version=2, final_seed_namespace=942107, development_seed_namespace=526103,
    production_revision='ab5baf0', production_pull_request='https://github.com/skirsch/KCOR/pull/4',
    predecessor='objection3_resolution/design.json', correction='Independent mc_id clock and sample-aware diagnostics')
(NEW/'design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
provenance = dict(reused_source_sha256=copied, source_snapshot_sha256={
    str(p.relative_to(NEW)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (NEW/'source_snapshot').rglob('*') if p.is_file()},
    original_module_unchanged=True, pr='https://github.com/skirsch/KCOR/pull/4')
(NEW/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
print(NEW)
