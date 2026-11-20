"""Restart only unfinished fixed-study jobs after a lost suspended process tree."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import psutil

HERE = Path(__file__).resolve().parents[1]
RUN = HERE/'final_2026-11-10'
sys.path.insert(0,str(HERE))
from run_study import hashes


def main():
    manifest = json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['phase'] == 'final'
    assert manifest['hashes'] == hashes(), 'Source or configuration changed'
    jsons = {(p.parent.name,p.stem) for p in RUN.glob('*/*.json') if len(p.stem)==4 and p.stem.isdigit()}
    npzs = {(p.parent.name,p.stem) for p in RUN.glob('*/*.npz') if len(p.stem)==4 and p.stem.isdigit()}
    assert jsons == npzs, 'Incomplete dataset file pair needs diagnosis'
    assert len(jsons) == 1062, f'Unexpected completed count: {len(jsons)}'
    assert not (RUN/'COMPLETE.json').exists()
    active = []
    for p in psutil.process_iter(['pid','name','cmdline']):
        if not (p.info['name'] or '').lower().startswith('python'):
            continue
        command = ' '.join(p.info['cmdline'] or []).replace('\\','/')
        if 'objection3_corrected_validation' in command and ('run_study.py' in command or 'execute_final.py' in command):
            active.append(p.info['pid'])
    assert not active, f'Study is already active: {active}'
    stages = [
        ('run',[str(HERE/'run_study.py'),'--phase','final','--out',str(RUN),'--workers','6','--resume']),
        ('summarize',[str(HERE/'summarize.py'),str(RUN)]),
        ('audit',[str(HERE/'finish_study.py'),str(RUN)])]
    results = []
    for stage,command in stages:
        with (HERE/f'final_resume_{stage}.log').open('x',encoding='utf-8') as log:
            proc = subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(stage=stage,exit_code=proc.returncode,finished=datetime.now(timezone.utc).isoformat()))
        print(stage, 'exit:', proc.returncode, flush=True)
        if proc.returncode:
            break
    with (HERE/'final_execution_resumed.json').open('x',encoding='utf-8') as f:
        json.dump(results,f,indent=2)
    if any(r['exit_code'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
