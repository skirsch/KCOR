from functools import partial
from pathlib import Path
from types import SimpleNamespace
import os, sys
os.environ['DATASET'] = 'Czech'
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
import KCOR
adapter = SimpleNamespace(K=KCOR)
production_fit = KCOR.fit_theta0_gompertz
joint_fit = partial(production_fit, joint_seed=True)
def initialize(weekly_update):
    assert weekly_update == 'revised'
