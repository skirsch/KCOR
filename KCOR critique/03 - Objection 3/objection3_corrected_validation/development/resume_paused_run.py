"""Resume this study only on the user's subsequent request to do so."""
import json
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
state_file = root/'pause_state.json'
state = json.loads(state_file.read_text(encoding='utf-8'))
if state.get('resumed_at_utc'):
    raise SystemExit('This recorded pause has already been resumed')
coordinator = psutil.Process(state['coordinator'])
runner = psutil.Process(state['runner'])
assert runner.ppid() == coordinator.pid
assert 'objection3_corrected_validation/execute_final.py' in ' '.join(coordinator.cmdline()).replace('\\', '/')
assert 'objection3_corrected_validation/run_study.py' in ' '.join(runner.cmdline()).replace('\\', '/')
workers = [psutil.Process(pid) for pid in state['workers']]
assert len(workers) == 6 and all(p.ppid() == runner.pid for p in workers)
assert all(p.status() == psutil.STATUS_STOPPED for p in [*workers, runner, coordinator])
for process in [*workers, runner, coordinator]:
    process.resume()
from datetime import datetime, timezone
state['resumed_at_utc'] = datetime.now(timezone.utc).isoformat()
state_file.write_text(json.dumps(state, indent=2), encoding='utf-8')
print(f"Resumed study runner {runner.pid} with {len(workers)} workers")
