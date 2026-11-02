import os
import sys
from pathlib import Path

# One thread for the numeric libraries, set before they load, so the run time of the tests does not depend on
# the machine's core count or load
N_THREADS = 1
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = str(N_THREADS)

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def fast_rules(monkeypatch):
    """The yeast rules and bootstraps with few rounds and resamples."""
    from deadahead.yeast import constants as C
    from deadahead.yeast import leadtime as LT
    from deadahead.yeast import lifespans as L
    from deadahead.yeast import rules as B
    monkeypatch.setattr(L, "NUM_ROUNDS", 10)
    monkeypatch.setattr(B.survival_cif, "__kwdefaults__", {**B.survival_cif.__kwdefaults__, "rounds": 10})
    monkeypatch.setattr(B.bootstrap, "__kwdefaults__", {"n_boot": 50, "seed": C.SEED})
    monkeypatch.setattr(LT.median_lead_bootstrap, "__kwdefaults__", {"n_boot": 50, "seed": C.SEED})


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    """Put PyTorch back on the tests' thread count after each test, so results do not depend on test order."""
    yield
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(N_THREADS)
