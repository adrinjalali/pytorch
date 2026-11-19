# Owner(s): ["module: unknown"]

import collections
import os
import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_utils import IS_LINUX, run_tests, TestCase


try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


device = torch.device("cpu")


class Network(torch.nn.Module):
    maxp1 = torch.nn.MaxPool2d(1, 1)

    def forward(self, x):
        return self.maxp1(x)


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
class TestOpenMP_ParallelFor(TestCase):
    batch = 20
    channels = 1
    side_dim = 80
    x = torch.randn([batch, channels, side_dim, side_dim], device=device)
    model = Network()

    def func(self, runs):
        p = psutil.Process()
        # warm up for 5 runs, then things should be stable for the last 5
        last_rss = collections.deque(maxlen=5)
        for _ in range(10):
            for _ in range(runs):
                self.model(self.x)
            last_rss.append(p.memory_info().rss)
        return last_rss

    def func_rss(self, runs):
        last_rss = list(self.func(runs))
        # Check that the sequence is not strictly increasing
        is_increasing = True
        for idx in range(len(last_rss)):
            if idx == 0:
                continue
            is_increasing = is_increasing and (last_rss[idx] > last_rss[idx - 1])
        self.assertTrue(
            not is_increasing,
            msg=lambda msg: f"{msg}\nmemory usage is increasing, {str(last_rss)}",
        )

    def test_one_thread(self):
        """Make sure there is no memory leak with one thread: issue gh-32284"""
        torch.set_num_threads(1)
        self.func_rss(300)

    def test_n_threads(self):
        """Make sure there is no memory leak with many threads"""
        ncores = min(5, psutil.cpu_count(logical=False))
        torch.set_num_threads(ncores)
        self.func_rss(300)


@unittest.skipIf(not IS_LINUX, "Tests OpenMP thread binding on Linux")
@unittest.skipIf(not torch.backends.openmp.is_available(), "Requires OpenMP")
class TestOpenMP_DefaultNumThreads(TestCase):
    def default_num_threads(self, **env):
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        }
        child_env.update(env)
        out = subprocess.check_output(
            [sys.executable, "-c", "import torch; print(torch.get_num_threads())"],
            env=child_env,
            text=True,
        )
        return int(out.strip().splitlines()[-1])

    def test_thread_binding_keeps_default_num_threads(self):
        """With OMP_PROC_BIND or OMP_PLACES set, the OpenMP runtime binds the
        initial thread to a single place as it loads. torch's default thread
        count must still follow the CPUs the process may use."""
        expected = self.default_num_threads()
        for env in ({"OMP_PROC_BIND": "true"}, {"OMP_PLACES": "cores"}):
            with self.subTest(**env):
                self.assertEqual(self.default_num_threads(**env), expected)


if __name__ == "__main__":
    run_tests()
