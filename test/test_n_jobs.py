import numpy as np
import pandas as pd
import psutil
import pytest
import torch

from harmony import harmonize


@pytest.mark.skipif(not hasattr(psutil.Process, "cpu_affinity"), reason="no CPU affinity on this platform")
def test_default_n_jobs_respects_cpu_affinity():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    df = pd.DataFrame({"batch": rng.choice(["a", "b"], size=200)})

    harmonize(X, df, batch_key="batch", max_iter_harmony=1, verbose=False)

    assert torch.get_num_threads() <= len(psutil.Process().cpu_affinity())
