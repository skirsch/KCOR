"""Replay the established checks against the exact isolated PR checkout."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here / 'verify_bundle.py').read_text(encoding='utf-8')
source = source.replace("out = HERE/'verification'", "out = HERE/'pr_verification'")
source = source.replace("tests = HERE/'submission/test/monte_carlo/test_sample_isolation.py'",
    "tests = Path('C:/Users/jasmi/repo/KCOR-mc-fix-pr/test/monte_carlo/test_sample_isolation.py')")
source = source.replace("HERE/'submission/code/KCOR.py'",
    "Path('C:/Users/jasmi/repo/KCOR-mc-fix-pr/code/KCOR.py')")
exec(compile(source, str(here / 'verify_bundle.py'), 'exec'),
     {'__file__': str(here / 'verify_bundle.py'), '__name__': '__main__'})
