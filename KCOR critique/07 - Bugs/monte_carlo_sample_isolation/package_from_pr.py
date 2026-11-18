"""Build a portable, revision-pinned bug package from the existing PR checkout.

Reads Git objects and retained verification; does not modify a repository or PR.
Existing package artifacts are preserved by exclusive creation.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
CHECKOUT = Path('C:/Users/jasmi/repo/KCOR-mc-fix-pr')
BASE = 'b639be2572e04c3fbc8d15e102c56b87295959c9'
HEAD = 'ab5baf07fad1b0b340e494d22bb39c13295a2441'
EXPECTED = ['code/KCOR.py', 'test/monte_carlo/README.md',
            'test/monte_carlo/test_sample_isolation.py']


def git(*args):
    return subprocess.check_output(['git', '-C', str(CHECKOUT), *args])


def main():
    assert git('rev-parse', 'HEAD').decode().strip() == HEAD
    changed = git('diff', '--name-only', BASE, HEAD).decode().splitlines()
    assert sorted(changed) == sorted(EXPECTED)
    note = git('show', HEAD + ':test/monte_carlo/README.md').decode('utf-8')
    assert note.rstrip('\n') == (HERE/'REVIEWER_NOTE.md').read_text(encoding='utf-8').rstrip('\n')
    # apply_patch may add a terminal blank line; keep the shipped copy byte exact.
    (HERE/'REVIEWER_NOTE.md').write_bytes(note.encode('utf-8'))
    git('diff', '--check', BASE, HEAD)
    artifacts = {
        'fix.patch': git('diff', '--binary', BASE, HEAD),
        'test_sample_isolation.py': git('show', HEAD + ':test/monte_carlo/test_sample_isolation.py'),
        'verification.json': (REVIEW/'objection3_production_fix/pr_verification/verification.json').read_bytes(),
    }
    verification = json.loads(artifacts['verification.json'])
    assert verification['checks'] == 'passed'
    for name, content in artifacts.items():
        with (HERE/name).open('xb') as stream:
            stream.write(content)
    # Verify patch identity against the exact head checkout without applying it.
    git('apply', '--reverse', '--check', str(HERE/'fix.patch'))
    paths = sorted(artifacts) + ['REVIEWER_NOTE.md', 'README.md', 'package_from_pr.py']
    report = dict(
        created=datetime.now(timezone.utc).isoformat(),
        pull_request='https://github.com/skirsch/KCOR/pull/4',
        pr_status_observed='open, unmerged', status_checked='2026-11-13',
        base=BASE, head=HEAD, changed_files=changed,
        reviewer_note_source='test/monte_carlo/README.md at PR head; verified against live connector response',
        verification_source='objection3_production_fix/pr_verification/verification.json',
        git_diff_check='passed', reverse_apply_check_at_head='passed',
        sha256={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in paths})
    with (HERE/'provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='sha256'}, indent=2))


if __name__ == '__main__':
    main()
